import { useState, useEffect, useCallback, useRef } from 'react'
import { JobStatus, getJobStatus } from '@/lib/api'

export interface UseJobStatusOptions {
  enableWebSocket?: boolean
  onStatusChange?: (status: JobStatus) => void
  onComplete?: (result: JobStatus) => void
  onError?: (error: string) => void
}

export const useJobStatus = (
  jobId: string | null,
  options: UseJobStatusOptions = {}
) => {
  const {
    enableWebSocket = true,
    onStatusChange,
    onComplete,
    onError
  } = options

  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const isActiveRef = useRef(true)
  const isPollingRef = useRef(false)
  const hasWebSocketRef = useRef(false)
  const lastPollingIntervalRef = useRef<number>(0)

  // 🚀 ADAPTIVE POLLING INTERVALS - Much smarter!
  const getPollingInterval = useCallback((status?: string) => {
    switch (status) {
      case 'pending':
        return 8000 // 8s - slower for queued jobs
      case 'processing':
        return 1500 // 1.5s - fast for active generation
      case 'completed':
      case 'failed':
      case 'cancelled':
        return 0 // Stop polling
      default:
        return 5000 // 5s default
    }
  }, [])

  // Update job status and trigger callbacks
  const updateJobStatus = useCallback(
    (status: JobStatus) => {
      console.log('📊 Job Status Update:', {
        job_id: status.job_id,
        status: status.status,
        progress: status.progress,
        started_at: status.started_at,
        started_at_type: typeof status.started_at,
        created_at: status.created_at,
        eta: status.eta
      })

      setJobStatus(status)
      onStatusChange?.(status)

      // Check if job is completed
      if (status.status === 'completed' && status.result_url) {
        onComplete?.(status)
        setIsLoading(false)
      } else if (status.status === 'failed') {
        setError(status.error_message || 'Job failed')
        onError?.(status.error_message || 'Job failed')
        setIsLoading(false)
      } else if (status.status === 'processing') {
        setIsLoading(true)
        setError(null)
      }
    },
    [onStatusChange, onComplete, onError]
  )

  // Stop polling with proper cleanup
  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
    isPollingRef.current = false
    lastPollingIntervalRef.current = 0
  }, [])

  // Cleanup all connections and intervals
  const cleanup = useCallback(() => {
    isActiveRef.current = false

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    hasWebSocketRef.current = false

    // Stop polling
    stopPolling()
  }, [stopPolling])

  // 🎯 SMART POLLING with adaptive intervals
  const startPolling = useCallback(
    (currentJobId: string) => {
      // Prevent multiple polling instances
      if (isPollingRef.current || pollIntervalRef.current) {
        console.log('Polling already active, skipping...')
        return
      }

      console.log(`🚀 Starting adaptive polling for job ${currentJobId}`)
      isPollingRef.current = true

      const poll = async () => {
        if (!isActiveRef.current || !isPollingRef.current) return

        try {
          console.log(`🔄 Polling job status for ${currentJobId}...`)
          const response = await getJobStatus(currentJobId)
          if (response.data && !response.error) {
            const newStatus = response.data
            console.log(
              `📡 Polling received status: ${newStatus.status} (${newStatus.progress}%)`
            )
            updateJobStatus(newStatus)

            // 🔄 ADAPTIVE INTERVAL ADJUSTMENT
            const newInterval = getPollingInterval(newStatus.status)
            const currentInterval = lastPollingIntervalRef.current

            // Stop polling if job is finished
            if (
              ['completed', 'failed', 'cancelled'].includes(newStatus.status)
            ) {
              console.log(`✅ Job ${currentJobId} finished, stopping polling`)
              stopPolling()
              return
            }

            // 🎯 Smart interval switching - only restart if interval needs to change
            if (newInterval !== currentInterval && newInterval > 0) {
              console.log(
                `⚡ Switching polling interval: ${currentInterval}ms → ${newInterval}ms (status: ${newStatus.status})`
              )

              // Clear current interval
              if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current)
              }

              // Set new interval
              lastPollingIntervalRef.current = newInterval
              pollIntervalRef.current = setInterval(poll, newInterval)
            }
          } else if (response.error) {
            console.error(`❌ Job status error: ${response.error}`)
            setError(response.error)
            onError?.(response.error)
            stopPolling()
          }
        } catch (err) {
          console.error('❌ Failed to poll job status:', err)
          setError('Failed to get job status')
          onError?.('Failed to get job status')
          stopPolling()
        }
      }

      // Initial poll
      poll()

      // Set initial interval based on current status (default to pending)
      const initialInterval = getPollingInterval('pending')
      lastPollingIntervalRef.current = initialInterval

      if (isActiveRef.current && isPollingRef.current && initialInterval > 0) {
        pollIntervalRef.current = setInterval(poll, initialInterval)
        console.log(`📊 Initial polling interval: ${initialInterval}ms`)
      }
    },
    [getPollingInterval, updateJobStatus, onError, stopPolling]
  )

  // WebSocket connection for real-time updates (preferred over polling)
  const connectWebSocket = useCallback(
    (currentJobId: string) => {
      if (!enableWebSocket || wsRef.current || hasWebSocketRef.current) {
        console.log('WebSocket already connected or disabled')
        return
      }

      console.log(`🔌 Connecting WebSocket for job ${currentJobId}`)
      const baseWsUrl =
        process.env.NEXT_PUBLIC_WS_URL ||
        `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${
          window.location.hostname
        }:8000`
      const wsUrl = `${baseWsUrl}/ws/${currentJobId}`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log(`✅ WebSocket connected for job ${currentJobId}`)
        hasWebSocketRef.current = true
        // Stop polling if WebSocket is working - WebSocket is much more efficient
        if (isPollingRef.current) {
          console.log('🔌 WebSocket connected, stopping polling for efficiency')
          stopPolling()
        }
      }

      ws.onmessage = (event) => {
        try {
          const status: JobStatus = JSON.parse(event.data)
          console.log(
            `🔌 WebSocket received status: ${status.status} (${status.progress}%)`
          )
          updateJobStatus(status)
        } catch (err) {
          console.error('❌ Failed to parse WebSocket message:', err)
        }
      }

      ws.onclose = () => {
        console.log(`🔌 WebSocket disconnected for job ${currentJobId}`)
        wsRef.current = null
        hasWebSocketRef.current = false
        // Fallback to adaptive polling only if job is still active
        if (isActiveRef.current && !isPollingRef.current) {
          console.log('🔄 WebSocket closed, falling back to adaptive polling')
          startPolling(currentJobId)
        }
      }

      ws.onerror = (wsError) => {
        console.error('❌ WebSocket error:', wsError)
        wsRef.current = null
        hasWebSocketRef.current = false
        // Fallback to adaptive polling only if not already polling
        if (isActiveRef.current && !isPollingRef.current) {
          console.log('🔄 WebSocket error, falling back to adaptive polling')
          startPolling(currentJobId)
        }
      }

      wsRef.current = ws
    },
    [enableWebSocket, updateJobStatus, startPolling, stopPolling]
  )

  // Start tracking when jobId changes
  useEffect(() => {
    if (!jobId) {
      cleanup()
      setJobStatus(null)
      setIsLoading(false)
      setError(null)
      return
    }

    console.log(`🎯 Starting smart job tracking for ${jobId}`)
    isActiveRef.current = true
    setIsLoading(true)
    setError(null)

    // Start with WebSocket if enabled (most efficient), otherwise use adaptive polling
    if (enableWebSocket) {
      connectWebSocket(jobId as string)
      // Fallback to adaptive polling after a delay only if WebSocket fails to connect
      const fallbackTimeout = setTimeout(() => {
        if (
          !hasWebSocketRef.current &&
          !isPollingRef.current &&
          isActiveRef.current
        ) {
          console.log(
            '⏰ WebSocket fallback timeout, starting adaptive polling'
          )
          startPolling(jobId as string)
        }
      }, 3000) // 3 second timeout

      return () => {
        clearTimeout(fallbackTimeout)
        cleanup()
      }
    }
    startPolling(jobId as string)
    return cleanup
  }, [jobId, enableWebSocket])

  // Cleanup on unmount
  useEffect(() => cleanup, [cleanup])

  // Manual refresh function
  const refresh = useCallback(async () => {
    if (!jobId) return

    try {
      const response = await getJobStatus(jobId)
      if (response.data && !response.error) {
        updateJobStatus(response.data)
      } else if (response.error) {
        setError(response.error)
        onError?.(response.error)
      }
    } catch (err) {
      console.error('❌ Failed to refresh job status:', err)
      setError('Failed to refresh job status')
      onError?.('Failed to refresh job status')
    }
  }, [jobId, updateJobStatus, onError])

  return {
    jobStatus,
    isLoading,
    error,
    refresh,
    cleanup
  }
}

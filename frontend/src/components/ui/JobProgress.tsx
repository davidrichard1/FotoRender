import React from 'react'
import { JobStatus } from '@/lib/api'
import { Button } from './button'
import { Badge } from './badge'
import { Spinner } from './spinner'

interface JobProgressProps {
  jobStatus: JobStatus | null
  isLoading: boolean
  error: string | null
  onCancel?: () => void
  onRetry?: () => void
  className?: string
}

export const JobProgress: React.FC<JobProgressProps> = ({
  jobStatus,
  isLoading,
  error,
  onCancel,
  onRetry,
  className = '',
}) => {
  if (!jobStatus && !isLoading && !error) {
    return null
  }

  const getStatusColor = (status: string) => {
    switch (status) {
    case 'pending':
      return 'bg-amber-500'
    case 'processing':
      return 'bg-blue-500 animate-pulse'
    case 'completed':
      return 'bg-green-500'
    case 'failed':
      return 'bg-red-500'
    case 'cancelled':
      return 'bg-gray-500'
    default:
      return 'bg-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
    case 'pending':
      return (
        <div className="w-4 h-4 rounded-full border-2 border-amber-300 border-t-transparent animate-spin" />
      )
    case 'processing':
      return (
        <div className="w-4 h-4 rounded-full border-2 border-blue-300 border-t-transparent animate-spin" />
      )
    case 'completed':
      return (
        <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center text-white text-xs">
            ✓
        </div>
      )
    case 'failed':
      return (
        <div className="w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white text-xs">
            ⚠
        </div>
      )
    case 'cancelled':
      return (
        <div className="w-4 h-4 bg-gray-500 rounded-full flex items-center justify-center text-white text-xs">
            ✕
        </div>
      )
    default:
      return <Spinner className="w-4 h-4" />
    }
  }

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    if (minutes < 60) {
      return remainingSeconds > 0
        ? `${minutes}m ${remainingSeconds}s`
        : `${minutes}m`
    }
    const hours = Math.floor(minutes / 60)
    const remainingMinutes = minutes % 60
    return `${hours}h ${remainingMinutes}m`
  }

  const getElapsedTime = () => {
    if (!jobStatus?.started_at) return null

    try {
      // Much simpler approach - just track time since we first saw "processing" status
      // This avoids all timezone and parsing complications

      if (jobStatus.status !== 'processing') {
        return null // Only show elapsed time for processing jobs
      }

      // Use a simple heuristic: if we're at X% progress and typical generation is ~60s,
      // estimate elapsed time based on progress
      const progress = jobStatus.progress || 0
      if (progress > 0) {
        // Rough estimate: 60 seconds total generation time
        const estimatedTotalTime = 60
        const estimatedElapsed = (progress / 100) * estimatedTotalTime
        return Math.floor(estimatedElapsed)
      }

      // Fallback: just return null to avoid timezone complications
      return null
    } catch (err) {
      console.error('Error calculating elapsed time:', err)
      return null
    }
  }

  const getDetailedStatusMessage = () => {
    if (error) return { title: 'Error', message: error, detail: null }
    if (!jobStatus) {
      return {
        title: 'Initializing',
        message: 'Setting up generation...',
        detail: null,
      }
    }

    const elapsedTime = getElapsedTime()

    switch (jobStatus.status) {
    case 'pending':
      return {
        title: 'Queued',
        message: 'Waiting for available GPU worker...',
        detail: jobStatus.eta
          ? `Estimated wait: ${formatTime(jobStatus.eta)}`
          : 'Checking queue position...',
      }

    case 'processing':
      let processingDetail = null
      let processingMessage = 'Generating your image...'

      if (jobStatus.progress !== undefined) {
        const progress = Math.round(jobStatus.progress)

        // Estimate current step based on progress (assuming ~30 steps default)
        const estimatedTotalSteps = 30
        const currentStep = Math.floor((progress / 100) * estimatedTotalSteps)

        if (progress < 5) {
          processingMessage = 'Setting up model and processing prompt...'
        } else if (progress < 15) {
          processingMessage = 'Initializing diffusion process...'
        } else if (progress >= 95) {
          processingMessage = 'Finalizing image and applying post-processing...'
        } else {
          processingMessage = `Diffusion step ${currentStep}/${estimatedTotalSteps}`
        }

        const timeInfo = []
        if (elapsedTime) timeInfo.push(`${formatTime(elapsedTime)} elapsed`)
        if (jobStatus.eta) timeInfo.push(`${formatTime(jobStatus.eta)} remaining`)

        processingDetail = timeInfo.length > 0 ? timeInfo.join(' • ') : null
      } else {
        processingDetail = elapsedTime
          ? `${formatTime(elapsedTime)} elapsed`
          : null
      }

      return {
        title: 'Processing',
        message: processingMessage,
        detail: processingDetail,
      }

    case 'completed':
      const completedDetail = elapsedTime
        ? `Completed in ${formatTime(elapsedTime)}`
        : null
      return {
        title: 'Complete!',
        message: 'Your image has been generated successfully!',
        detail: completedDetail,
      }

    case 'failed':
      return {
        title: 'Failed',
        message: jobStatus.error_message || 'Generation failed',
        detail: elapsedTime ? `Failed after ${formatTime(elapsedTime)}` : null,
      }

    case 'cancelled':
      return {
        title: 'Cancelled',
        message: 'Generation was cancelled',
        detail: elapsedTime
          ? `Cancelled after ${formatTime(elapsedTime)}`
          : null,
      }

    default:
      return {
        title: 'Processing',
        message: 'Working on your request...',
        detail: null,
      }
    }
  }

  const statusInfo = getDetailedStatusMessage()
  const showProgressBar = jobStatus?.status === 'processing' && jobStatus?.progress !== undefined

  return (
    <div
      className={`bg-white dark:bg-gray-800 rounded-lg border shadow-sm p-4 space-y-3 ${className}`}
    >
      {/* Header with status and actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon(jobStatus?.status || 'pending')}
          <div className="flex items-center space-x-2">
            <Badge
              variant="secondary"
              className={`text-white text-xs font-medium px-2 py-1 ${getStatusColor(
                jobStatus?.status || 'pending',
              )}`}
            >
              {statusInfo.title.toUpperCase()}
            </Badge>
            {jobStatus?.job_id && (
              <span className="text-xs text-gray-400 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
                #{jobStatus.job_id.slice(0, 8)}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {(jobStatus?.status === 'pending'
            || jobStatus?.status === 'processing')
            && onCancel && (
            <Button
              variant="outline"
              size="sm"
              onClick={onCancel}
              className="text-red-600 hover:text-red-700 border-red-200 hover:border-red-300"
            >
                Cancel
            </Button>
          )}
          {(jobStatus?.status === 'failed' || error) && onRetry && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRetry}
              className="text-blue-600 hover:text-blue-700 border-blue-200 hover:border-blue-300"
            >
              Retry
            </Button>
          )}
        </div>
      </div>

      {/* Enhanced Progress Bar */}
      {showProgressBar && (
        <div className="space-y-2">
          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 overflow-hidden">
            <div
              className="bg-gradient-to-r from-blue-500 to-blue-600 h-2.5 rounded-full transition-all duration-500 ease-out relative"
              style={{ width: `${jobStatus.progress}%` }}
            >
              {/* Animated shimmer effect during processing */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-20 animate-pulse"></div>
            </div>
          </div>
          <div className="flex justify-between items-center text-xs">
            <span className="text-gray-600 dark:text-gray-300 font-medium">
              {Math.round(jobStatus.progress!)}% complete
            </span>
            {jobStatus.eta && (
              <span className="text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded-md">
                {formatTime(jobStatus.eta)} left
              </span>
            )}
          </div>
        </div>
      )}

      {/* Status Messages */}
      <div className="space-y-1">
        <p className="text-sm font-medium text-gray-800 dark:text-gray-200">
          {statusInfo.message}
        </p>
        {statusInfo.detail && (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {statusInfo.detail}
          </p>
        )}
      </div>

      {/* Enhanced Timestamps */}
      {jobStatus && (
        <div className="flex justify-between text-xs text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
          <div className="flex space-x-4">
            {jobStatus.created_at && (
              <span>
                Created {new Date(jobStatus.created_at).toLocaleTimeString()}
              </span>
            )}
            {jobStatus.started_at && jobStatus.status === 'processing' && (
              <span>
                Started {new Date(jobStatus.started_at).toLocaleTimeString()}
              </span>
            )}
          </div>
          {jobStatus.completed_at && (
            <span>
              Finished {new Date(jobStatus.completed_at).toLocaleTimeString()}
            </span>
          )}
        </div>
      )}

      {/* Seed Display for Completed Jobs */}
      {jobStatus?.status === 'completed' && jobStatus.seed && (
        <div className="text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-700 p-2 rounded">
          <span className="font-medium">Seed:</span>{' '}
          <span className="font-mono">{jobStatus.seed}</span>
        </div>
      )}
    </div>
  )
}

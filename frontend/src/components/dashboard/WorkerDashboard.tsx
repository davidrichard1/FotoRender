import React, { useState, useEffect } from 'react'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Spinner } from '../ui/spinner'
import { Card } from '../ui/card'

interface Worker {
  id: string
  type: 'local' | 'cloud'
  status: 'running' | 'stopped' | 'error'
  gpu_id?: number
  jobs_processed: number
  current_job?: string
  started_at: string
  last_seen?: string
}

interface QueueStats {
  pending_jobs: number
  processing_jobs: number
  completed_jobs: number
  failed_jobs: number
  workers_active: number
  average_wait_time?: number
}

interface WorkerDashboardData {
  workers: Record<string, Worker>
  queue_stats: QueueStats
  total_workers: number
  active_workers: number
}

export const WorkerDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<WorkerDashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isStartingWorker, setIsStartingWorker] = useState(false)

  // Fetch worker data
  const fetchWorkerData = async () => {
    try {
      const response = await fetch('http://localhost:8000/workers')
      if (!response.ok) {
        throw new Error(`Failed to fetch workers: ${response.status}`)
      }
      const data = await response.json()
      setDashboardData(data)
      setError(null)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to fetch worker data',
      )
    } finally {
      setIsLoading(false)
    }
  }

  // Start a new worker
  const startWorker = async (
    type: 'local' | 'cloud' = 'local',
    gpuId: number = 0,
  ) => {
    setIsStartingWorker(true)
    try {
      const response = await fetch('http://localhost:8000/workers/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ worker_type: type, gpu_id: gpuId }),
      })

      if (!response.ok) {
        throw new Error(`Failed to start worker: ${response.status}`)
      }

      // Refresh data
      await fetchWorkerData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start worker')
    } finally {
      setIsStartingWorker(false)
    }
  }

  // Stop a worker
  const stopWorker = async (workerId: string) => {
    try {
      const response = await fetch(
        `http://localhost:8000/workers/${workerId}`,
        {
          method: 'DELETE',
        },
      )

      if (!response.ok) {
        throw new Error(`Failed to stop worker: ${response.status}`)
      }

      // Refresh data
      await fetchWorkerData()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop worker')
    }
  }

  // Auto-refresh every 5 seconds
  useEffect(() => {
    fetchWorkerData()
    const interval = setInterval(fetchWorkerData, 5000)
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
    case 'running':
      return 'bg-green-500'
    case 'stopped':
      return 'bg-gray-500'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-yellow-500'
    }
  }

  const getStatusEmoji = (status: string) => {
    switch (status) {
    case 'running':
      return '🟢'
    case 'stopped':
      return '🔴'
    case 'error':
      return '❌'
    default:
      return '🟡'
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Spinner className="w-8 h-8" />
        <span className="ml-2">Loading worker dashboard...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
        <h3 className="text-red-800 dark:text-red-200 font-semibold mb-2">
          Error Loading Workers
        </h3>
        <p className="text-red-600 dark:text-red-300 mb-4">{error}</p>
        <Button onClick={fetchWorkerData} variant="outline" size="sm">
          Retry
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            🔧 Worker Management Dashboard
          </h2>
          <p className="text-gray-600 dark:text-gray-300 mt-1">
            Monitor and manage your GPU workers for scalable image generation
          </p>
        </div>

        <div className="flex space-x-2">
          <Button
            onClick={() => startWorker('local', 0)}
            disabled={isStartingWorker}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {isStartingWorker ? (
              <>
                <Spinner className="w-4 h-4 mr-2" />
                Starting...
              </>
            ) : (
              '🚀 Start Local Worker'
            )}
          </Button>

          <Button onClick={fetchWorkerData} variant="outline" size="sm">
            🔄 Refresh
          </Button>
        </div>
      </div>

      {/* Queue Statistics */}
      {dashboardData && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
              {dashboardData.queue_stats.pending_jobs}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Pending Jobs
            </div>
          </Card>

          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {dashboardData.queue_stats.processing_jobs}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Processing
            </div>
          </Card>

          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {dashboardData.queue_stats.completed_jobs}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Completed
            </div>
          </Card>

          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {dashboardData.queue_stats.failed_jobs}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Failed
            </div>
          </Card>

          <Card className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
              {dashboardData.active_workers}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300">
              Active Workers
            </div>
          </Card>
        </div>
      )}

      {/* Workers List */}
      {dashboardData && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Workers ({Object.keys(dashboardData.workers).length})
          </h3>

          {Object.keys(dashboardData.workers).length === 0 ? (
            <Card className="p-8 text-center">
              <div className="text-gray-500 dark:text-gray-400 mb-4">
                <div className="text-6xl mb-4">🤖</div>
                <h3 className="text-lg font-semibold mb-2">
                  No Workers Running
                </h3>
                <p className="mb-4">
                  Start your first worker to begin processing image generation
                  jobs.
                </p>
                <Button
                  onClick={() => startWorker('local', 0)}
                  disabled={isStartingWorker}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  🚀 Start Your First Worker
                </Button>
              </div>
            </Card>
          ) : (
            <div className="grid gap-4">
              {Object.values(dashboardData.workers).map((worker) => (
                <Card key={worker.id} className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="text-2xl">
                        {getStatusEmoji(worker.status)}
                      </div>

                      <div>
                        <div className="flex items-center space-x-2">
                          <h4 className="font-semibold text-gray-900 dark:text-white">
                            {worker.id}
                          </h4>
                          <Badge
                            variant="secondary"
                            className={`text-white ${getStatusColor(
                              worker.status,
                            )}`}
                          >
                            {worker.status.toUpperCase()}
                          </Badge>
                          <Badge variant="outline">
                            {worker.type.toUpperCase()}
                          </Badge>
                        </div>

                        <div className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                          <span>Jobs Processed: {worker.jobs_processed}</span>
                          {worker.gpu_id !== undefined && (
                            <span className="ml-4">GPU: {worker.gpu_id}</span>
                          )}
                          {worker.current_job && (
                            <span className="ml-4">
                              Current: {worker.current_job.slice(0, 8)}...
                            </span>
                          )}
                        </div>

                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                          Started:{' '}
                          {new Date(worker.started_at).toLocaleString()}
                        </div>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      {worker.status === 'running' && (
                        <Button
                          onClick={() => stopWorker(worker.id)}
                          variant="destructive"
                          size="sm"
                        >
                          🛑 Stop
                        </Button>
                      )}

                      {worker.status === 'stopped' && (
                        <Button
                          onClick={() => startWorker(worker.type, worker.gpu_id)
                          }
                          variant="outline"
                          size="sm"
                        >
                          ▶️ Restart
                        </Button>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Performance Insights */}
      {dashboardData?.queue_stats.average_wait_time && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            📊 Performance Insights
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-blue-600 dark:text-blue-400">
                {dashboardData.queue_stats.average_wait_time}s
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Average Wait Time
              </div>
            </div>
            <div>
              <div className="text-xl font-bold text-green-600 dark:text-green-400">
                {Math.round(
                  (dashboardData.queue_stats.completed_jobs
                    / (dashboardData.queue_stats.completed_jobs
                      + dashboardData.queue_stats.failed_jobs))
                    * 100 || 0,
                )}
                %
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Success Rate
              </div>
            </div>
            <div>
              <div className="text-xl font-bold text-purple-600 dark:text-purple-400">
                {dashboardData.active_workers > 0
                  ? Math.round(
                    (dashboardData.queue_stats.processing_jobs
                        / dashboardData.active_workers)
                        * 100,
                  ) / 100
                  : 0}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Jobs per Worker
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
}

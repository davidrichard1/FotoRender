import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/components/ui/toast'
import { imageCache } from '@/lib/imageCache'

interface CacheStats {
  totalImages: number
  totalSize: number
  memoryUsage: number
  indexedDBUsage: number
}

export function CacheManager() {
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [loading, setLoading] = useState(false)
  const { showSuccess, showError } = useToast()

  const loadStats = async () => {
    try {
      setLoading(true)
      const cacheStats = await imageCache.getCacheStats()
      setStats(cacheStats)
    } catch (error) {
      console.error('Failed to load cache stats:', error)
      showError('Failed to load cache statistics')
    } finally {
      setLoading(false)
    }
  }

  const handleClearCache = async () => {
    try {
      setLoading(true)
      await imageCache.clearCache()
      await loadStats()
      showSuccess('Cache cleared successfully')
    } catch (error) {
      console.error('Failed to clear cache:', error)
      showError('Failed to clear cache')
    } finally {
      setLoading(false)
    }
  }

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
  }

  useEffect(() => {
    loadStats()
  }, [])

  return (
    <Card className="bg-[#161B22] border-[#30363D] p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-white">Image Cache Management</h3>
          <p className="text-gray-400 text-sm mt-1">
            Monitor and manage the image caching system
          </p>
        </div>
        <Badge variant="secondary" className="bg-[#00D4AA]/20 text-[#00D4AA]">
          Active
        </Badge>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#00D4AA]"></div>
        </div>
      ) : stats ? (
        <div className="space-y-6">
          {/* Cache Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
              <div className="text-2xl font-bold text-white">{stats.totalImages}</div>
              <div className="text-sm text-gray-400">Cached Images</div>
            </div>
            
            <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
              <div className="text-2xl font-bold text-white">{formatBytes(stats.totalSize)}</div>
              <div className="text-sm text-gray-400">Total Size</div>
            </div>
            
            <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
              <div className="text-2xl font-bold text-white">{formatBytes(stats.memoryUsage)}</div>
              <div className="text-sm text-gray-400">Memory Cache</div>
            </div>
            
            <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
              <div className="text-2xl font-bold text-white">{formatBytes(stats.indexedDBUsage)}</div>
              <div className="text-sm text-gray-400">IndexedDB Cache</div>
            </div>
          </div>

          {/* Cache Distribution */}
          <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
            <h4 className="text-white font-medium mb-3">Cache Distribution</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">Memory Cache</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-[#00D4AA] h-2 rounded-full" 
                      style={{ 
                        width: `${stats.totalSize > 0 ? (stats.memoryUsage / stats.totalSize) * 100 : 0}%` 
                      }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-400 min-w-[3rem]">
                    {stats.totalSize > 0 ? Math.round((stats.memoryUsage / stats.totalSize) * 100) : 0}%
                  </span>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-gray-400 text-sm">IndexedDB Cache</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full" 
                      style={{ 
                        width: `${stats.totalSize > 0 ? (stats.indexedDBUsage / stats.totalSize) * 100 : 0}%` 
                      }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-400 min-w-[3rem]">
                    {stats.totalSize > 0 ? Math.round((stats.indexedDBUsage / stats.totalSize) * 100) : 0}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-wrap gap-3">
            <Button
              onClick={loadStats}
              disabled={loading}
              variant="outline"
              className="flex items-center gap-2"
            >
              🔄 Refresh Stats
            </Button>
            
            <Button
              onClick={handleClearCache}
              disabled={loading}
              variant="outline"
              className="flex items-center gap-2 hover:bg-red-500/20 hover:border-red-500/50 hover:text-red-400"
            >
              🗑️ Clear Cache
            </Button>
          </div>

          {/* Cache Info */}
          <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4">
            <h4 className="text-white font-medium mb-2">Cache Configuration</h4>
            <div className="text-sm text-gray-400 space-y-1">
              <div>• Memory Cache Limit: 50 MB</div>
              <div>• IndexedDB Cache Limit: 200 MB</div>
              <div>• Cache Expiry: 24 hours</div>
              <div>• Automatic background preloading enabled</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-400">
          Failed to load cache statistics
        </div>
      )}
    </Card>
  )
} 
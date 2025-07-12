'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  Button, Card, CardContent, Badge,
} from '@/components/ui'

interface Upscaler {
  id: string
  filename: string
  displayName: string
  description?: string
  upscalerType: 'REALESRGAN' | 'ESRGAN' | 'WAIFU2X' | 'OTHER'
  scaleFactor: number
  isActive: boolean
  isDefault: boolean
  fileSize?: number
  author?: string
  version?: string
  website?: string
  tags: string[]
  usageCount: number
  lastUsed?: string
  createdAt: string
  updatedAt: string
}

interface UpscalersTableProps {
  onAdd: () => void
  onEdit: (upscaler: Upscaler) => void
  onDelete: (upscaler: Upscaler) => void
}

export function UpscalersTable({
  onAdd,
  onEdit,
  onDelete,
}: UpscalersTableProps) {
  // Helper function to avoid nested ternary expressions
  const getUpscalerTypeBadgeClassName = (upscalerType: string) => {
    switch (upscalerType) {
    case 'REALESRGAN':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
    case 'ESRGAN':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    case 'WAIFU2X':
      return 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const [upscalers, setUpscalers] = useState<Upscaler[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchUpscalers = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
      const response = await fetch(`http://${apiHost}/upscalers`)

      if (!response.ok) {
        throw new Error(`Failed to fetch Upscalers: ${response.statusText}`)
      }

      const data = await response.json()
      setUpscalers(data.upscalers || [])
    } catch (err) {
      console.error('Error fetching Upscalers:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch Upscalers')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchUpscalers()
  }, [fetchUpscalers])

  const handleDelete = (upscaler: Upscaler) => {
    // Use the prop-based delete handler
    onDelete(upscaler)
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">
                Loading Upscalers...
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center py-12">
            <div className="text-4xl mb-4">❌</div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Error Loading Upscalers
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <Button onClick={fetchUpscalers} variant="outline">
              🔄 Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Upscaler Models
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Manage upscaler models for enhancing image resolution and quality.
          </p>
        </div>
        <Button onClick={onAdd} className="bg-purple-600 hover:bg-purple-700">
          ➕ Add Upscaler
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Upscalers
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {upscalers.length}
                </p>
              </div>
              <div className="text-2xl">📈</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Active
                </p>
                <p className="text-2xl font-bold text-green-600">
                  {upscalers.filter((u) => u.isActive).length}
                </p>
              </div>
              <div className="text-2xl">✅</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  RealESRGAN
                </p>
                <p className="text-2xl font-bold text-blue-600">
                  {
                    upscalers.filter((u) => u.upscalerType === 'REALESRGAN')
                      .length
                  }
                </p>
              </div>
              <div className="text-2xl">🎯</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Avg Scale
                </p>
                <p className="text-2xl font-bold text-purple-600">
                  {upscalers.length > 0
                    ? `${(
                      upscalers.reduce((sum, u) => sum + u.scaleFactor, 0)
                        / upscalers.length
                    ).toFixed(1)}x`
                    : '0x'}
                </p>
              </div>
              <div className="text-2xl">📊</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Table */}
      <Card>
        <CardContent className="p-6">
          {upscalers.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">📈</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                No Upscalers Found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Get started by adding your first upscaler model to enhance image
                resolution.
              </p>
              <Button
                onClick={onAdd}
                className="bg-purple-600 hover:bg-purple-700"
              >
                ➕ Add Your First Upscaler
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Scale Factor
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Usage
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                  {upscalers.map((upscaler) => (
                    <tr
                      key={upscaler.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-gray-900 dark:text-white">
                            {upscaler.displayName}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {upscaler.filename}
                          </span>
                          {upscaler.description && (
                            <span className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                              {upscaler.description}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Badge
                          className={getUpscalerTypeBadgeClassName(
                            upscaler.upscalerType,
                          )}
                        >
                          {upscaler.upscalerType}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className="font-mono text-lg font-bold text-purple-600 dark:text-purple-400">
                          {upscaler.scaleFactor}x
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Badge
                            className={
                              upscaler.isActive
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            }
                          >
                            {upscaler.isActive ? '✅ Active' : '❌ Inactive'}
                          </Badge>
                          {upscaler.isDefault && (
                            <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                              ⭐ Default
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className="font-mono text-sm">
                          {upscaler.usageCount}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm font-mono">
                          {upscaler.fileSize
                            ? `${(upscaler.fileSize / (1024 * 1024)).toFixed(
                              1,
                            )} MB`
                            : 'Unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onEdit(upscaler)}
                          >
                            ✏️ Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleDelete(upscaler)}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20"
                          >
                            🗑️ Delete
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  Button, Card, CardContent, Badge,
} from '@/components/ui'

interface Sampler {
  id: string
  name: string
  displayName: string
  description?: string
  category: 'EULER' | 'DPM' | 'DDIM' | 'PLMS' | 'KARRAS' | 'OTHER'
  isActive: boolean
  isDefault: boolean
  steps: { min: number; max: number; recommended: number }
  speed: 'FAST' | 'MEDIUM' | 'SLOW'
  quality: 'LOW' | 'MEDIUM' | 'HIGH'
  usageCount: number
  lastUsed?: string
  createdAt: string
  updatedAt: string
}

interface SamplersTableProps {
  onAdd: () => void
  onEdit: (sampler: Sampler) => void
  onDelete: (sampler: Sampler) => void
}

export function SamplersTable({ onAdd, onEdit, onDelete }: SamplersTableProps) {
  // Helper functions to avoid nested ternary expressions
  const getCategoryBadgeClassName = (category: string) => {
    switch (category) {
    case 'EULER':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
    case 'DPM':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    case 'DDIM':
      return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
    case 'KARRAS':
      return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getSpeedBadgeClassName = (speed: string) => {
    switch (speed) {
    case 'FAST':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    case 'MEDIUM':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    default:
      return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
    }
  }

  const getSpeedBadgeText = (speed: string) => {
    switch (speed) {
    case 'FAST':
      return '⚡ Fast'
    case 'MEDIUM':
      return '🚀 Medium'
    default:
      return '🐌 Slow'
    }
  }

  const getQualityBadgeClassName = (quality: string) => {
    switch (quality) {
    case 'HIGH':
      return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
    case 'MEDIUM':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getQualityBadgeText = (quality: string) => {
    switch (quality) {
    case 'HIGH':
      return '💎 High'
    case 'MEDIUM':
      return '🔸 Medium'
    default:
      return '🔹 Low'
    }
  }

  const [samplers, setSamplers] = useState<Sampler[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSamplers = useCallback(async () => {
    try {
      setLoading(true)
      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
      const response = await fetch(`http://${apiHost}/samplers`)

      if (!response.ok) {
        throw new Error(`Failed to fetch Samplers: ${response.statusText}`)
      }

      const data = await response.json()
      setSamplers(data.samplers || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching Samplers:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch Samplers')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchSamplers()
  }, [fetchSamplers])

  const handleDelete = (sampler: Sampler) => {
    // Use the prop-based delete handler
    onDelete(sampler)
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">
                Loading Samplers...
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
              Error Loading Samplers
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <Button onClick={fetchSamplers} variant="outline">
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
            Sampling Methods
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Manage sampling algorithms for controlling generation quality and
            speed.
          </p>
        </div>
        <Button onClick={onAdd} className="bg-green-600 hover:bg-green-700">
          ➕ Add Sampler
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Samplers
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {samplers.length}
                </p>
              </div>
              <div className="text-2xl">🎲</div>
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
                  {samplers.filter((s) => s.isActive).length}
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
                  Fast Samplers
                </p>
                <p className="text-2xl font-bold text-blue-600">
                  {samplers.filter((s) => s.speed === 'FAST').length}
                </p>
              </div>
              <div className="text-2xl">⚡</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  High Quality
                </p>
                <p className="text-2xl font-bold text-purple-600">
                  {samplers.filter((s) => s.quality === 'HIGH').length}
                </p>
              </div>
              <div className="text-2xl">💎</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Table */}
      <Card>
        <CardContent className="p-6">
          {samplers.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🎲</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                No Samplers Found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Get started by adding your first sampling method to control
                generation quality.
              </p>
              <Button
                onClick={onAdd}
                className="bg-green-600 hover:bg-green-700"
              >
                ➕ Add Your First Sampler
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
                      Category
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Recommended Steps
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Performance
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-center text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Usage
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                  {samplers.map((sampler) => (
                    <tr
                      key={sampler.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-gray-900 dark:text-white">
                            {sampler.displayName}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {sampler.name}
                          </span>
                          {sampler.description && (
                            <span className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                              {sampler.description}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Badge
                          className={getCategoryBadgeClassName(
                            sampler.category,
                          )}
                        >
                          {sampler.category}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <div className="flex flex-col items-center">
                          <span className="font-mono text-lg font-bold text-green-600 dark:text-green-400">
                            {sampler.steps.recommended}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            ({sampler.steps.min}-{sampler.steps.max})
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Badge
                            className={getSpeedBadgeClassName(sampler.speed)}
                          >
                            {getSpeedBadgeText(sampler.speed)}
                          </Badge>
                          <Badge
                            className={getQualityBadgeClassName(
                              sampler.quality,
                            )}
                          >
                            {getQualityBadgeText(sampler.quality)}
                          </Badge>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Badge
                            className={
                              sampler.isActive
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            }
                          >
                            {sampler.isActive ? '✅ Active' : '❌ Inactive'}
                          </Badge>
                          {sampler.isDefault && (
                            <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                              ⭐ Default
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className="font-mono text-sm">
                          {sampler.usageCount}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onEdit(sampler)}
                          >
                            ✏️ Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleDelete(sampler)}
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

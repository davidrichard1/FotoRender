'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  Button, Card, CardContent, Badge,
} from '@/components/ui'

interface SystemConfig {
  id: string
  key: string
  value: string
  description?: string
  category:
    | 'GENERATION'
    | 'STORAGE'
    | 'PERFORMANCE'
    | 'UI'
    | 'SECURITY'
    | 'OTHER'
  dataType: 'STRING' | 'INTEGER' | 'FLOAT' | 'BOOLEAN' | 'JSON'
  isDefault: boolean
  isEditable: boolean
  validation?: string
  lastModified?: string
  modifiedBy?: string
  createdAt: string
  updatedAt: string
}

interface SystemConfigTableProps {
  onAdd: () => void
  onEdit: (config: SystemConfig) => void
  onDelete: (config: SystemConfig) => void
}

export function SystemConfigTable({
  onAdd,
  onEdit,
  onDelete,
}: SystemConfigTableProps) {
  // Helper function to avoid nested ternary expressions
  const getCategoryBadgeClassName = (category: string) => {
    switch (category) {
    case 'GENERATION':
      return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
    case 'PERFORMANCE':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
    case 'STORAGE':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    case 'UI':
      return 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200'
    case 'SECURITY':
      return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const [configs, setConfigs] = useState<SystemConfig[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchConfig = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
      const response = await fetch(`http://${apiHost}/config`)

      if (!response.ok) {
        throw new Error(`Failed to fetch System Config: ${response.statusText}`)
      }

      const data = await response.json()
      setConfigs(data.configs || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching System Config:', err)
      setError(
        err instanceof Error ? err.message : 'Failed to fetch System Config',
      )
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  const handleDelete = (config: SystemConfig) => {
    // Use the prop-based delete handler
    onDelete(config)
  }

  const renderValue = (config: SystemConfig) => {
    const { value } = config
    const maxLength = 50

    if (config.dataType === 'BOOLEAN') {
      return (
        <Badge
          className={
            value === 'true'
              ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
              : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
          }
        >
          {value === 'true' ? '✅ True' : '❌ False'}
        </Badge>
      )
    }

    if (config.dataType === 'JSON') {
      try {
        const parsed = JSON.parse(value)
        const formatted = JSON.stringify(parsed, null, 2)
        return (
          <div className="font-mono text-xs">
            {formatted.length > maxLength
              ? `${formatted.substring(0, maxLength)}...`
              : formatted}
          </div>
        )
      } catch {
        return (
          <span className="text-red-500 font-mono text-xs">Invalid JSON</span>
        )
      }
    }

    return (
      <span className="font-mono text-sm">
        {value.length > maxLength
          ? `${value.substring(0, maxLength)}...`
          : value}
      </span>
    )
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">
                Loading System Configuration...
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
              Error Loading System Configuration
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <Button onClick={fetchConfig} variant="outline">
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
            System Configuration
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Manage global settings and configuration parameters for the AI
            generation system.
          </p>
        </div>
        <Button onClick={onAdd} className="bg-orange-600 hover:bg-orange-700">
          ➕ Add Config
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total Settings
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {configs.length}
                </p>
              </div>
              <div className="text-2xl">⚙️</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Editable
                </p>
                <p className="text-2xl font-bold text-green-600">
                  {configs.filter((c) => c.isEditable).length}
                </p>
              </div>
              <div className="text-2xl">✏️</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Performance
                </p>
                <p className="text-2xl font-bold text-blue-600">
                  {configs.filter((c) => c.category === 'PERFORMANCE').length}
                </p>
              </div>
              <div className="text-2xl">🚀</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Generation
                </p>
                <p className="text-2xl font-bold text-purple-600">
                  {configs.filter((c) => c.category === 'GENERATION').length}
                </p>
              </div>
              <div className="text-2xl">🎨</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Table */}
      <Card>
        <CardContent className="p-6">
          {configs.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">⚙️</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                No Configuration Found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Get started by adding your first system configuration setting.
              </p>
              <Button
                onClick={onAdd}
                className="bg-orange-600 hover:bg-orange-700"
              >
                ➕ Add Your First Config
              </Button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Key
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Value
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Category
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Last Modified
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                  {configs.map((config) => (
                    <tr
                      key={config.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-gray-900 dark:text-white">
                            {config.key}
                          </span>
                          {config.description && (
                            <span className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                              {config.description}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4">{renderValue(config)}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Badge
                          className={getCategoryBadgeClassName(config.category)}
                        >
                          {config.category}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Badge className="bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200">
                          {config.dataType}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Badge
                            className={
                              config.isEditable
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                            }
                          >
                            {config.isEditable ? '✏️ Editable' : '🔒 Read-Only'}
                          </Badge>
                          {config.isDefault && (
                            <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                              🔧 Default
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          {config.lastModified && (
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                              {new Date(
                                config.lastModified,
                              ).toLocaleDateString()}
                            </span>
                          )}
                          {config.modifiedBy && (
                            <span className="text-xs text-gray-500 dark:text-gray-500">
                              by {config.modifiedBy}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onEdit(config)}
                            disabled={!config.isEditable}
                          >
                            ✏️ Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleDelete(config)}
                            disabled={!config.isEditable}
                            className="text-red-600 hover:text-red-700 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-900/20 disabled:opacity-50 disabled:cursor-not-allowed"
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

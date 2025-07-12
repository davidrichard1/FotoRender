'use client'

import React, { useState, useEffect, useCallback } from 'react'
import {
  Button, Card, CardContent, Badge,
} from '@/components/ui'

interface Vae {
  id: string
  filename: string
  displayName: string
  description?: string
  vaeType: 'SDXL' | 'SD15' | 'GENERAL'
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

interface VaesTableProps {
  onAdd: () => void
  onEdit: (vae: Vae) => void
  onDelete: (vae: Vae) => void
}

export function VaesTable({ onAdd, onEdit, onDelete }: VaesTableProps) {
  // Helper function to avoid nested ternary expressions
  const getVaeTypeBadgeClassName = (vaeType: string) => {
    switch (vaeType) {
    case 'SDXL':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
    case 'SD15':
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const [vaes, setVaes] = useState<Vae[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchVaes = useCallback(async () => {
    try {
      setLoading(true)
      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
      const response = await fetch(`http://${apiHost}/vaes`)

      if (!response.ok) {
        throw new Error(`Failed to fetch VAEs: ${response.statusText}`)
      }

      const data = await response.json()
      setVaes(data.vaes || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching VAEs:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch VAEs')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchVaes()
  }, [fetchVaes])

  const handleDelete = (vae: Vae) => {
    // Use the prop-based delete handler
    onDelete(vae)
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-600 dark:text-gray-400">
                Loading VAEs...
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
              Error Loading VAEs
            </h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
            <Button onClick={fetchVaes} variant="outline">
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
            VAE Models
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Manage Variational Autoencoders for improved image quality and style
            consistency.
          </p>
        </div>
        <Button onClick={onAdd} className="bg-blue-600 hover:bg-blue-700">
          ➕ Add VAE
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total VAEs
                </p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                  {vaes.length}
                </p>
              </div>
              <div className="text-2xl">🖼️</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Active VAEs
                </p>
                <p className="text-2xl font-bold text-green-600">
                  {vaes.filter((v) => v.isActive).length}
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
                  SDXL VAEs
                </p>
                <p className="text-2xl font-bold text-blue-600">
                  {vaes.filter((v) => v.vaeType === 'SDXL').length}
                </p>
              </div>
              <div className="text-2xl">🎨</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Default VAE
                </p>
                <p className="text-lg font-bold text-yellow-600">
                  {vaes.find((v) => v.isDefault)?.displayName || 'None'}
                </p>
              </div>
              <div className="text-2xl">⭐</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Table */}
      <Card>
        <CardContent className="p-6">
          {vaes.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-6xl mb-4">🖼️</div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                No VAEs Found
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Get started by adding your first VAE model to enhance image
                generation quality.
              </p>
              <Button onClick={onAdd} className="bg-blue-600 hover:bg-blue-700">
                ➕ Add Your First VAE
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
                      Description
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Type
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
                  {vaes.map((vae) => (
                    <tr
                      key={vae.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex flex-col">
                          <span className="font-medium text-gray-900 dark:text-white">
                            {vae.displayName}
                          </span>
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            {vae.filename}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-sm text-gray-600 dark:text-gray-400 max-w-xs truncate block">
                          {vae.description || 'No description'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <Badge
                          className={getVaeTypeBadgeClassName(vae.vaeType)}
                        >
                          {vae.vaeType}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Badge
                            className={
                              vae.isActive
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                            }
                          >
                            {vae.isActive ? '✅ Active' : '❌ Inactive'}
                          </Badge>
                          {vae.isDefault && (
                            <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200">
                              ⭐ Default
                            </Badge>
                          )}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-center">
                        <span className="font-mono text-sm">
                          {vae.usageCount}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm font-mono">
                          {vae.fileSize
                            ? `${(vae.fileSize / (1024 * 1024)).toFixed(1)} MB`
                            : 'Unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => onEdit(vae)}
                          >
                            ✏️ Edit
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleDelete(vae)}
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

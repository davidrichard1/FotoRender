'use client'

import React, { useState } from 'react'
import {
  Button,
  Card,
  CardContent,
  Badge,
  ConfirmDialog
} from '@/components/ui'
import { useToast } from '@/components/ui/toast'
import { ModelsTable } from '@/components/admin/ModelsTable'
import { LorasTable } from '@/components/admin/LorasTable'
import { VaesTable } from '@/components/admin/VaesTable'
import { UpscalersTable } from '@/components/admin/UpscalersTable'
import { SamplersTable } from '@/components/admin/SamplersTable'
import { SystemConfigTable } from '@/components/admin/SystemConfigTable'
import { CacheManager } from '@/components/admin/CacheManager'
import CompatibilityManager from '@/components/admin/CompatibilityManager'
import ModelModal from '@/components/admin/ModelModal'
import LoraModal from '@/components/admin/LoraModal'
import { WorkerDashboard } from '@/components/dashboard/WorkerDashboard'
import { ApiModel, ApiLora } from '@/lib/api'

// Import interfaces from table components
interface LoRa {
  id: string
  filename: string
  displayName: string
  description?: string
  category: string
  isGated: boolean
  isActive: boolean
  compatibleBases: string[]
  usageCount: number
  lastUsed?: string
  fileSize: number
  createdAt: string
  updatedAt: string
}

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

// Define the UI model type (matches ModelsTable's AiModel interface)
interface AiModel {
  id: string
  filename: string
  displayName: string
  description?: string
  modelType: 'SDXL' | 'SD15' | 'FLUX' | 'OTHER'
  baseModel: string
  isGated: boolean
  isActive: boolean
  author?: string
  version?: string
  tags: string[]
  usageCount: number
  lastUsed?: string
  createdAt: string
  updatedAt: string
}

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState('overview')

  // Modal states for models
  const [isModelModalOpen, setIsModelModalOpen] = useState(false)
  const [editingModel, setEditingModel] = useState<ApiModel | null>(null)

  // Modal states for LoRAs
  const [isLoraModalOpen, setIsLoraModalOpen] = useState(false)
  const [editingLora, setEditingLora] = useState<ApiLora | null>(null)

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean
    title: string
    message: string
    onConfirm: () => void
  }>({
    isOpen: false,
    title: '',
    message: '',
    onConfirm: () => {}
  })

  // Toast notifications
  const { showError, showSuccess, ToastContainer } = useToast()

  // Helper function to show confirmation dialog
  const showConfirmDialog = (
    title: string,
    message: string,
    onConfirm: () => void
  ) => {
    setConfirmDialog({
      isOpen: true,
      title,
      message,
      onConfirm
    })
  }

  const closeConfirmDialog = () => {
    setConfirmDialog((prev) => ({ ...prev, isOpen: false }))
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'workers', label: 'Workers', icon: '🔧' },
    { id: 'models', label: 'AI Models', icon: '🤖' },
    { id: 'loras', label: 'LoRAs', icon: '🎯' },
    { id: 'compatibility', label: 'Compatibility', icon: '🔗' },
    { id: 'vaes', label: 'VAEs', icon: '🖼️' },
    { id: 'upscalers', label: 'Upscalers', icon: '📈' },
    { id: 'samplers', label: 'Samplers', icon: '🎲' },
    { id: 'cache', label: 'Image Cache', icon: '💾' },
    { id: 'system', label: 'System Config', icon: '⚙️' }
  ]

  const handleAddModel = () => {
    setEditingModel(null)
    setIsModelModalOpen(true)
  }

  const handleSyncModels = async () => {
    try {
      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
        
      const response = await fetch(`http://${apiHost}/models/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      if (result.success) {
        showSuccess(`Successfully synced ${result.details.synced} new models from filesystem`)
        
        // Trigger a page refresh to show new models
        window.location.reload()
      } else {
        throw new Error(result.message || 'Sync failed')
      }
    } catch (error) {
      console.error('Failed to sync models:', error)
      showError(error instanceof Error ? error.message : 'Failed to sync models from filesystem')
    }
  }

  const handleEditModel = (model: AiModel) => {
    // Convert the UI model back to API model format for editing
    const apiModel: ApiModel = {
      filename: model.filename,
      display_name: model.displayName,
      path: `models/${model.filename}`,
      type: model.modelType.toLowerCase(),
      capability: 'text-to-image',
      source: 'local',
      base_model: model.baseModel.toLowerCase().replace('_', '-'),
      description: model.description || '',
      is_gated: model.isGated,
      file_size_mb: null,
      usage_count: model.usageCount,
      last_used: model.lastUsed || null
    }
    setEditingModel(apiModel)
    setIsModelModalOpen(true)
  }

  const handleDeleteModel = async (model: AiModel) => {
    const performDelete = async () => {
      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const response = await fetch(
          `http://${apiHost}/models/${model.filename}`,
          {
            method: 'DELETE'
          }
        )

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Failed to delete AI Model')
        }

        console.log('AI Model deleted successfully:', model.filename)
        showSuccess(`AI Model "${model.displayName}" deleted successfully`)
        closeConfirmDialog()
        // The table will automatically refresh due to cache invalidation
      } catch (error) {
        console.error('Error deleting AI Model:', error)
        showError(
          `Failed to delete AI Model: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        )
        closeConfirmDialog()
      }
    }

    showConfirmDialog(
      'Delete AI Model',
      `Are you sure you want to delete the AI Model "${model.displayName}"? This action cannot be undone.`,
      performDelete
    )
  }

  const handleModelSuccess = (model: ApiModel) => {
    console.log('Model operation successful:', model)
    // The table will automatically refresh due to cache invalidation
    setIsModelModalOpen(false)
    setEditingModel(null)
  }

  const handleAddLora = () => {
    setEditingLora(null)
    setIsLoraModalOpen(true)
  }

  const handleEditLora = (lora: LoRa) => {
    // Convert the UI LoRA to API LoRA format for editing
    const apiLora: ApiLora = {
      filename: lora.filename,
      display_name: lora.displayName,
      description: lora.description || '',
      category: lora.category,
      is_gated: lora.isGated,
      trigger_words: [],
      default_scale: 1.0,
      min_scale: -5.0,
      max_scale: 5.0,
      recommended_min: 0.5,
      recommended_max: 1.5,
      usage_tips: '',
      author: '',
      version: '',
      website: '',
      tags: [],
      usage_count: lora.usageCount || 0,
      last_used: lora.lastUsed || null,
      file_size: lora.fileSize || undefined,
      source: 'local'
    }
    setEditingLora(apiLora)
    setIsLoraModalOpen(true)
  }

  const handleDeleteLora = async (lora: LoRa) => {
    const performDelete = async () => {
      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const response = await fetch(
          `http://${apiHost}/loras/${lora.filename}`,
          {
            method: 'DELETE'
          }
        )

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Failed to delete LoRA')
        }

        console.log('LoRA deleted successfully:', lora.filename)
        showSuccess(`LoRA "${lora.displayName}" deleted successfully`)
        closeConfirmDialog()
        // The table will automatically refresh due to cache invalidation
      } catch (error) {
        console.error('Error deleting LoRA:', error)
        showError(
          `Failed to delete LoRA: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        )
        closeConfirmDialog()
      }
    }

    showConfirmDialog(
      'Delete LoRA',
      `Are you sure you want to delete the LoRA "${lora.displayName}"?`,
      performDelete
    )
  }

  const handleLoraSuccess = (lora: ApiLora) => {
    console.log('LoRA operation successful:', lora)
    // The table will automatically refresh due to cache invalidation
    setIsLoraModalOpen(false)
    setEditingLora(null)
  }

  // Delete handlers for other components
  const handleDeleteVae = async (vae: Vae) => {
    const performDelete = async () => {
      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const response = await fetch(`http://${apiHost}/vaes/${vae.filename}`, {
          method: 'DELETE'
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Failed to delete VAE')
        }

        console.log('VAE deleted successfully:', vae.filename)
        showSuccess(`VAE "${vae.displayName}" deleted successfully`)
        closeConfirmDialog()
      } catch (error) {
        console.error('Error deleting VAE:', error)
        showError(
          `Failed to delete VAE: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        )
        closeConfirmDialog()
      }
    }

    showConfirmDialog(
      'Delete VAE',
      `Are you sure you want to delete the VAE "${vae.displayName}"? This action cannot be undone.`,
      performDelete
    )
  }

  const handleDeleteUpscaler = async (upscaler: Upscaler) => {
    if (
      !confirm(
        `Are you sure you want to delete the Upscaler "${upscaler.displayName}"? This action cannot be undone.`
      )
    ) {
      return
    }

    try {
      const apiHost =
        typeof window !== 'undefined'
          ? `${window.location.hostname}:8000`
          : 'localhost:8000'
      const response = await fetch(
        `http://${apiHost}/upscalers/${upscaler.filename}`,
        {
          method: 'DELETE'
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to delete Upscaler')
      }

      console.log('Upscaler deleted successfully:', upscaler.filename)
    } catch (error) {
      console.error('Error deleting Upscaler:', error)
      alert(
        `Failed to delete Upscaler: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      )
    }
  }

  const handleDeleteSampler = async (sampler: Sampler) => {
    if (
      !confirm(
        `Are you sure you want to delete the Sampler "${sampler.displayName}"? This action cannot be undone.`
      )
    ) {
      return
    }

    try {
      const apiHost =
        typeof window !== 'undefined'
          ? `${window.location.hostname}:8000`
          : 'localhost:8000'
      const response = await fetch(
        `http://${apiHost}/samplers/${sampler.name}`,
        {
          method: 'DELETE'
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to delete Sampler')
      }

      console.log('Sampler deleted successfully:', sampler.name)
    } catch (error) {
      console.error('Error deleting Sampler:', error)
      alert(
        `Failed to delete Sampler: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`
      )
    }
  }

  const handleDeleteConfig = async (config: SystemConfig) => {
    if (!config.isEditable) {
      showError('This configuration item cannot be deleted.')
      return
    }

    const performDelete = async () => {
      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const response = await fetch(`http://${apiHost}/config/${config.key}`, {
          method: 'DELETE'
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.error || 'Failed to delete config')
        }

        console.log('Config deleted successfully:', config.key)
        showSuccess(`Config "${config.key}" deleted successfully`)
        closeConfirmDialog()
      } catch (error) {
        console.error('Error deleting config:', error)
        showError(
          `Failed to delete config: ${
            error instanceof Error ? error.message : 'Unknown error'
          }`
        )
        closeConfirmDialog()
      }
    }

    showConfirmDialog(
      'Delete Configuration',
      `Are you sure you want to delete the config "${config.key}"? This action cannot be undone.`,
      performDelete
    )
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-6">
            {/* Admin Welcome Banner */}
            <Card className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border-blue-200 dark:border-blue-800">
              <CardContent className="p-6">
                <div className="flex items-center gap-4">
                  <div className="text-4xl">🎨</div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                      Foto Render AI - Admin Panel
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400">
                      Manage AI models, LoRAs, compatibility settings, and
                      system configuration for image generation.
                    </p>
                  </div>
                  <div className="ml-auto">
                    <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                      🛡️ Admin Access
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        AI Models
                      </p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        7
                      </p>
                    </div>
                    <div className="text-2xl">🤖</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        LoRAs
                      </p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        12
                      </p>
                    </div>
                    <div className="text-2xl">🎯</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        Compatibility Links
                      </p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        --
                      </p>
                    </div>
                    <div className="text-2xl">🔗</div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        Total Assets
                      </p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        25+
                      </p>
                    </div>
                    <div className="text-2xl">📊</div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Quick Actions
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Button
                    variant="outline"
                    className="h-20 flex flex-col gap-2 hover:bg-blue-50 hover:border-blue-300 dark:hover:bg-blue-900/20"
                    onClick={() => setActiveTab('models')}
                  >
                    <span className="text-2xl">🤖</span>
                    <span>Manage Models</span>
                  </Button>
                  <Button
                    variant="outline"
                    className="h-20 flex flex-col gap-2 hover:bg-purple-50 hover:border-purple-300 dark:hover:bg-purple-900/20"
                    onClick={() => setActiveTab('loras')}
                  >
                    <span className="text-2xl">🎯</span>
                    <span>Manage LoRAs</span>
                  </Button>
                  <Button
                    variant="outline"
                    className="h-20 flex flex-col gap-2 hover:bg-green-50 hover:border-green-300 dark:hover:bg-green-900/20"
                    onClick={() => setActiveTab('compatibility')}
                  >
                    <span className="text-2xl">🔗</span>
                    <span>Set Compatibility</span>
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  System Status
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Database Connection
                    </span>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      ✅ Connected
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      AI Generation Service
                    </span>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      ✅ Online
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      File Storage
                    </span>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      ✅ Available
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      Model Files
                    </span>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      ✅ Synced
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )

      case 'workers':
        return <WorkerDashboard />

      case 'models':
        return (
          <ModelsTable
            onAdd={handleAddModel}
            onEdit={handleEditModel}
            onDelete={handleDeleteModel}
            onSync={handleSyncModels}
          />
        )

      case 'loras':
        return (
          <LorasTable
            onAdd={handleAddLora}
            onEdit={handleEditLora}
            onDelete={handleDeleteLora}
          />
        )

      case 'compatibility':
        return <CompatibilityManager />

      case 'vaes':
        return (
          <VaesTable
            onAdd={() => alert('VAE Add functionality coming soon!')}
            onEdit={(vae) =>
              alert(`Edit VAE: ${vae.displayName} - Coming soon!`)
            }
            onDelete={handleDeleteVae}
          />
        )

      case 'upscalers':
        return (
          <UpscalersTable
            onAdd={() => alert('Upscaler Add functionality coming soon!')}
            onEdit={(upscaler) =>
              alert(`Edit Upscaler: ${upscaler.displayName} - Coming soon!`)
            }
            onDelete={handleDeleteUpscaler}
          />
        )

      case 'samplers':
        return (
          <SamplersTable
            onAdd={() => alert('Sampler Add functionality coming soon!')}
            onEdit={(sampler) =>
              alert(`Edit Sampler: ${sampler.displayName} - Coming soon!`)
            }
            onDelete={handleDeleteSampler}
          />
        )

      case 'cache':
        return <CacheManager />

      case 'system':
        return (
          <SystemConfigTable
            onAdd={() => alert('System Config Add functionality coming soon!')}
            onEdit={(config) =>
              alert(`Edit Config: ${config.key} - Coming soon!`)
            }
            onDelete={handleDeleteConfig}
          />
        )

      default:
        return (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">
              {tabs.find((t) => t.id === activeTab)?.icon}
            </div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              {tabs.find((t) => t.id === activeTab)?.label}
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              This section is coming soon. We will implement comprehensive
              management for{' '}
              {tabs.find((t) => t.id === activeTab)?.label.toLowerCase()}.
            </p>
            <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
              🚧 In Development
            </Badge>
          </div>
        )
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-[1400px] mx-auto p-6">
        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-3xl">⚙️</span>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
              Admin Panel
            </h1>
          </div>
          <p className="text-gray-600 dark:text-gray-400">
            Manage AI models, LoRAs, compatibility settings, and system
            configuration for Foto Render AI.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="-mb-px flex space-x-8 overflow-x-auto">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                  }`}
                >
                  <span className="mr-2">{tab.icon}</span>
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        <div className="min-h-[600px]">{renderTabContent()}</div>
      </div>

      {/* Model Modal */}
      <ModelModal
        isOpen={isModelModalOpen}
        onClose={() => {
          setIsModelModalOpen(false)
          setEditingModel(null)
        }}
        model={editingModel}
        onSuccess={handleModelSuccess}
      />

      {/* LoRA Modal */}
      <LoraModal
        isOpen={isLoraModalOpen}
        onClose={() => {
          setIsLoraModalOpen(false)
          setEditingLora(null)
        }}
        lora={editingLora}
        onSuccess={handleLoraSuccess}
      />

      {/* Confirmation Dialog */}
      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        title={confirmDialog.title}
        message={confirmDialog.message}
        variant="destructive"
        onConfirm={() => {
          confirmDialog.onConfirm()
        }}
        onCancel={closeConfirmDialog}
      />

      {/* Toast Notifications */}
      <ToastContainer />
    </div>
  )
}

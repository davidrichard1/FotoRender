'use client'

import { useState, useEffect } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Spinner } from '@/components/ui/spinner'
import { Label } from '@/components/ui/label'
import { Select, SelectItem } from '@/components/ui/select'
import { apiClient } from '@/lib/api'
import type { ApiModel, ApiLora } from '@/lib/api'

interface CompatibilityData {
  lora_filename: string
  lora_display_name: string
  compatible_models: Array<{
    model_id: string
    filename: string
    display_name: string
    base_model: string
  }>
  total_compatible: number
}

export default function CompatibilityManager() {
  const [models, setModels] = useState<ApiModel[]>([])
  const [loras, setLoras] = useState<ApiLora[]>([])
  const [selectedLora, setSelectedLora] = useState<string>('')
  const [compatibility, setCompatibility] = useState<CompatibilityData | null>(
    null,
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Load models and LoRAs on component mount
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)

      const [modelsResponse, lorasResponse] = await Promise.all([
        apiClient.getModels(),
        apiClient.getLoras(),
      ])

      setModels(modelsResponse.data?.models || [])
      setLoras(lorasResponse.data?.loras || [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  // Load compatibility for selected LoRA
  const loadCompatibility = async (loraFilename: string) => {
    if (!loraFilename) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(
        `http://localhost:8000/loras/${loraFilename}/compatibility`,
      )
      if (!response.ok) {
        throw new Error('Failed to load compatibility data')
      }

      const data = await response.json()
      setCompatibility(data)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to load compatibility',
      )
      setCompatibility(null)
    } finally {
      setLoading(false)
    }
  }

  // Handle LoRA selection change
  const handleLoraChange = (loraFilename: string) => {
    setSelectedLora(loraFilename)
    setCompatibility(null)
    setSuccess(null)
    if (loraFilename) {
      loadCompatibility(loraFilename)
    }
  }

  // Add compatibility relationship
  const addCompatibility = async (modelFilename: string) => {
    if (!selectedLora) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(
        `http://localhost:8000/loras/${selectedLora}/compatibility/${modelFilename}`,
        {
          method: 'POST',
        },
      )

      if (!response.ok) {
        throw new Error('Failed to add compatibility')
      }

      const result = await response.json()
      setSuccess(result.message)

      // Reload compatibility data
      await loadCompatibility(selectedLora)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to add compatibility',
      )
    } finally {
      setLoading(false)
    }
  }

  // Remove compatibility relationship
  const removeCompatibility = async (modelFilename: string) => {
    if (!selectedLora) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(
        `http://localhost:8000/loras/${selectedLora}/compatibility/${modelFilename}`,
        {
          method: 'DELETE',
        },
      )

      if (!response.ok) {
        throw new Error('Failed to remove compatibility')
      }

      const result = await response.json()
      setSuccess(result.message)

      // Reload compatibility data
      await loadCompatibility(selectedLora)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to remove compatibility',
      )
    } finally {
      setLoading(false)
    }
  }

  // Set bulk compatibility
  const setBulkCompatibility = async (modelFilenames: string[]) => {
    if (!selectedLora) return

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(
        `http://localhost:8000/loras/${selectedLora}/compatibility/bulk`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model_filenames: modelFilenames,
          }),
        },
      )

      if (!response.ok) {
        throw new Error('Failed to set bulk compatibility')
      }

      const result = await response.json()
      setSuccess(result.message)

      // Reload compatibility data
      await loadCompatibility(selectedLora)
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to set bulk compatibility',
      )
    } finally {
      setLoading(false)
    }
  }

  // Auto-assign compatibility based on base model
  const autoAssignCompatibility = async () => {
    if (!selectedLora || !compatibility) return

    const selectedLoraData = loras.find((l) => l.filename === selectedLora)
    if (!selectedLoraData) return

    // Simple heuristic: assign to models with similar base model names
    const compatibleModels = models.filter((model) => {
      const loraName = selectedLoraData.display_name.toLowerCase()
      const modelBase = model.base_model.toLowerCase()

      // Check for direct matches
      if (loraName.includes('noobai') && modelBase.includes('noobai')) return true
      if (loraName.includes('pony') && modelBase.includes('pony')) return true
      if (loraName.includes('illustrious') && modelBase.includes('illustrious')) return true

      // Default to SDXL compatibility for generic LoRAs
      return modelBase.includes('sdxl') || modelBase.includes('base')
    })

    const modelFilenames = compatibleModels.map((m) => m.filename)
    await setBulkCompatibility(modelFilenames)
  }

  const isModelCompatible = (modelFilename: string): boolean => (
    compatibility?.compatible_models.some(
      (m) => m.filename === modelFilename,
    ) || false
  )

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🔗 LoRA-Model Compatibility Manager
          </CardTitle>
          <CardDescription>
            Manage which models are compatible with each LoRA for proper
            filtering on the generation page.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* LoRA Selection */}
          <div className="space-y-2">
            <Label htmlFor="lora-select">Select LoRA to Configure</Label>
            <Select
              value={selectedLora}
              onChange={(e) => handleLoraChange(e.target.value)}
            >
              <option value="">Choose a LoRA...</option>
              {loras.map((lora) => (
                <SelectItem key={lora.filename} value={lora.filename}>
                  {lora.display_name} ({lora.filename})
                </SelectItem>
              ))}
            </Select>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Spinner className="mr-2" />
              <span>Loading...</span>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          {/* Success Display */}
          {success && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-md">
              <p className="text-sm text-green-600">{success}</p>
            </div>
          )}

          {/* Compatibility Configuration */}
          {compatibility && selectedLora && !loading && (
            <>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">
                  Compatibility for: {compatibility.lora_display_name}
                </h3>
                <div className="space-x-2">
                  <Button
                    onClick={autoAssignCompatibility}
                    variant="outline"
                    size="sm"
                  >
                    🤖 Auto-Assign
                  </Button>
                  <Badge variant="secondary">
                    {compatibility.total_compatible} compatible models
                  </Badge>
                </div>
              </div>

              {/* Quick Actions */}
              <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
                <CardContent className="p-4">
                  <h4 className="font-semibold mb-3">Quick Actions</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    <Button
                      onClick={() => setBulkCompatibility(
                        models
                          .filter((m) => m.base_model.includes('noobai'))
                          .map((m) => m.filename),
                      )
                      }
                      variant="outline"
                      size="sm"
                      className="text-xs"
                    >
                      🎯 NoobAI Models Only
                    </Button>
                    <Button
                      onClick={() => setBulkCompatibility(
                        models
                          .filter((m) => m.base_model.includes('pony'))
                          .map((m) => m.filename),
                      )
                      }
                      variant="outline"
                      size="sm"
                      className="text-xs"
                    >
                      🐴 Pony Models Only
                    </Button>
                    <Button
                      onClick={() => setBulkCompatibility(
                        models
                          .filter((m) => m.base_model.includes('illustrious'))
                          .map((m) => m.filename),
                      )
                      }
                      variant="outline"
                      size="sm"
                      className="text-xs"
                    >
                      🎨 Illustrious Models Only
                    </Button>
                    <Button
                      onClick={() => setBulkCompatibility(models.map((m) => m.filename))
                      }
                      variant="outline"
                      size="sm"
                      className="text-xs"
                    >
                      ✅ All Models
                    </Button>
                    <Button
                      onClick={() => setBulkCompatibility([])}
                      variant="outline"
                      size="sm"
                      className="text-xs text-red-600"
                    >
                      🚫 Clear All
                    </Button>
                    <Button
                      onClick={() => setBulkCompatibility(
                        models
                          .filter((m) => !m.is_gated)
                          .map((m) => m.filename),
                      )
                      }
                      variant="outline"
                      size="sm"
                      className="text-xs"
                    >
                      🔒 SFW Models Only
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Available Models */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      Available Models
                    </CardTitle>
                    <CardDescription>
                      Click to add compatibility
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-60 overflow-y-auto ultra-smooth-scroll">
                      {models
                        .filter((model) => !isModelCompatible(model.filename))
                        .map((model) => (
                          <div
                            key={model.filename}
                            className="flex items-center justify-between p-2 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800"
                          >
                            <div className="flex-1 min-w-0">
                              <div className="font-medium text-sm truncate">
                                {model.display_name}
                              </div>
                              <div className="text-xs text-gray-500">
                                {model.base_model} • {model.model_type}
                              </div>
                            </div>
                            <Button
                              onClick={() => addCompatibility(model.filename)}
                              size="sm"
                              variant="outline"
                              disabled={loading}
                            >
                              Add
                            </Button>
                          </div>
                        ))}
                      {models.filter(
                        (model) => !isModelCompatible(model.filename),
                      ).length === 0 && (
                        <div className="text-center py-4 text-gray-500">
                          All models are already compatible
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Compatible Models */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      Compatible Models
                    </CardTitle>
                    <CardDescription>
                      Click to remove compatibility
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 max-h-60 overflow-y-auto ultra-smooth-scroll">
                      {compatibility.compatible_models.map((model) => (
                        <div
                          key={model.filename}
                          className="flex items-center justify-between p-2 border rounded-lg bg-green-50 dark:bg-green-900/20"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="font-medium text-sm truncate">
                              {model.display_name}
                            </div>
                            <div className="text-xs text-gray-500">
                              {model.base_model} • {model.filename}
                            </div>
                          </div>
                          <Button
                            onClick={() => removeCompatibility(model.filename)}
                            size="sm"
                            variant="outline"
                            disabled={loading}
                          >
                            Remove
                          </Button>
                        </div>
                      ))}
                      {compatibility.compatible_models.length === 0 && (
                        <div className="text-center py-4 text-gray-500">
                          No compatible models assigned
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          )}

          {/* Instructions */}
          {!selectedLora && (
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
              <p className="text-sm text-blue-600">
                Select a LoRA above to configure its model compatibility. This
                determines which models the LoRA will be available for on the
                generation page.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

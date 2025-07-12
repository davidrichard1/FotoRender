'use client'

import { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Textarea, Select, SelectItem, Switch,
} from '@/components/ui'
import { Badge } from '@/components/ui/badge'
import { ApiModel, createModel, updateModel } from '@/lib/api'

interface ModelModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: (model: ApiModel) => void
  model?: ApiModel | null // null for add, model for edit
}

interface ModelFormData {
  filename: string
  display_name: string
  type: string
  base_model: string
  file_format: string
  capability: string
  source: string
  description: string
  is_nsfw: boolean
  file_size_mb: number | null
}

// Model types for the main Models table (checkpoints/base models only)
const MODEL_TYPES = [
  { value: 'checkpoint', label: 'Checkpoint' },
  { value: 'base_model', label: 'Base Model' },
  { value: 'fine_tune', label: 'Fine-tuned Model' },
  { value: 'merge', label: 'Model Merge' },
  { value: 'custom', label: 'Custom Model' },
] as const

const BASE_MODELS = [
  // SD 1.x
  { value: 'sd_1_4', label: 'SD 1.4' },
  { value: 'sd_1_5', label: 'SD 1.5' },
  { value: 'sd_1_5_lcm', label: 'SD 1.5 LCM' },
  { value: 'sd_1_5_hyper', label: 'SD 1.5 Hyper' },

  // SD 2.x
  { value: 'sd_2_0', label: 'SD 2.0' },
  { value: 'sd_2_1', label: 'SD 2.1' },

  // SDXL
  { value: 'sdxl_1_0', label: 'SDXL 1.0' },
  { value: 'sd_3', label: 'SD 3' },
  { value: 'sd_3_5', label: 'SD 3.5' },
  { value: 'sd_3_5_medium', label: 'SD 3.5 Medium' },
  { value: 'sd_3_5_large', label: 'SD 3.5 Large' },
  { value: 'sd_3_5_large_turbo', label: 'SD 3.5 Large Turbo' },

  // Specialized
  { value: 'pony', label: 'Pony' },
  { value: 'flux_1_s', label: 'Flux.1 S' },
  { value: 'flux_1_d', label: 'Flux.1 D' },
  { value: 'flux_1_kontext', label: 'Flux.1 Kontext' },
  { value: 'aura_flow', label: 'Aura Flow' },
  { value: 'sdxl_lightning', label: 'SDXL Lightning' },
  { value: 'sdxl_hyper', label: 'SDXL Hyper' },
  { value: 'svd', label: 'SVD' },
  { value: 'pixart_alpha', label: 'PixArt α' },
  { value: 'pixart_sigma', label: 'PixArt Σ' },
  { value: 'hunyuan_1', label: 'Hunyuan 1' },
  { value: 'hunyuan_video', label: 'Hunyuan Video' },
  { value: 'lumina', label: 'Lumina' },
  { value: 'kolors', label: 'Kolors' },
  { value: 'illustrious', label: 'Illustrious' },
  { value: 'mochi', label: 'Mochi' },
  { value: 'ltxv', label: 'LTXV' },
  { value: 'cogvideox', label: 'CogVideoX' },
  { value: 'noobai', label: 'NoobAI' },
  { value: 'wan_video_1_38_b2v', label: 'Wan Video 1.38 B2v' },
  { value: 'wan_video_14b_b2v', label: 'Wan Video 14B B2v' },
  { value: 'wan_video_14b_b2v_480p', label: 'Wan Video 14B B2v 480p' },
  { value: 'wan_video_14b_i2v_720p', label: 'Wan Video 14B i2v 720p' },
  { value: 'hidream', label: 'HiDream' },
  { value: 'other', label: 'Other' },
] as const

const FILE_FORMATS = [
  { value: 'safetensors', label: 'SafeTensors' },
  { value: 'pickletensor', label: 'PickleTensor' },
  { value: 'gguf', label: 'GGUF' },
  { value: 'diffusers', label: 'Diffusers' },
  { value: 'core_ml', label: 'Core ML' },
  { value: 'onnx', label: 'ONNX' },
] as const

const CAPABILITIES = [
  { value: 'text-to-image', label: 'Text to Image' },
  { value: 'img2img', label: 'Image to Image' },
  { value: 'inpainting', label: 'Inpainting' },
  { value: 'outpainting', label: 'Outpainting' },
  { value: 'video-generation', label: 'Video Generation' },
  { value: 'general', label: 'General Purpose' },
] as const

const SOURCES = [
  { value: 'local', label: 'Local File' },
  { value: 'civitai', label: 'CivitAI' },
  { value: 'huggingface', label: 'Hugging Face' },
  { value: 'github', label: 'GitHub' },
  { value: 'custom', label: 'Custom URL' },
  { value: 'manual_upload', label: 'Manual Upload' },
] as const

export default function ModelModal({
  isOpen,
  onClose,
  onSuccess,
  model,
}: ModelModalProps) {
  const [formData, setFormData] = useState<ModelFormData>({
    filename: '',
    display_name: '',
    type: 'checkpoint',
    base_model: 'sdxl_1_0',
    file_format: 'safetensors',
    capability: 'text-to-image',
    source: 'local',
    description: '',
    is_nsfw: false,
    file_size_mb: null,
  })

  const [isLoading, setIsLoading] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [submitError, setSubmitError] = useState<string>('')

  const isEditing = !!model

  // Reset form when modal opens/closes or model changes
  useEffect(() => {
    if (isOpen) {
      if (model) {
        // Editing - populate form with model data
        setFormData({
          filename: model.filename,
          display_name: model.display_name,
          type: model.type,
          base_model: model.base_model,
          file_format: 'safetensors', // Default since API doesn't provide this yet
          capability: model.capability,
          source: model.source,
          description: model.description || '',
          is_nsfw: model.is_nsfw,
          file_size_mb: model.file_size_mb,
        })
      } else {
        // Adding - reset to defaults
        setFormData({
          filename: '',
          display_name: '',
          type: 'checkpoint',
          base_model: 'sdxl_1_0',
          file_format: 'safetensors',
          capability: 'text-to-image',
          source: 'local',
          description: '',
          is_nsfw: false,
          file_size_mb: null,
        })
      }
      setErrors({})
      setSubmitError('')
    }
  }, [isOpen, model])

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.filename.trim()) {
      newErrors.filename = 'Filename is required'
    } else if (!formData.filename.endsWith('.safetensors')) {
      newErrors.filename = 'Filename must end with .safetensors'
    }

    if (!formData.display_name.trim()) {
      newErrors.display_name = 'Display name is required'
    }

    if (formData.file_size_mb !== null && formData.file_size_mb <= 0) {
      newErrors.file_size_mb = 'File size must be positive'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    setIsLoading(true)
    setSubmitError('')

    try {
      let result

      if (isEditing && model) {
        // Update existing model
        result = await updateModel(model.filename, formData)
      } else {
        // Create new model
        result = await createModel(formData)
      }

      if (result.error) {
        setSubmitError(result.error)
      } else if (result.data) {
        // Success - call onSuccess with the model data
        onSuccess(result.data)
        onClose()
      }
    } catch (error) {
      setSubmitError(
        error instanceof Error ? error.message : 'An unexpected error occurred',
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleInputChange = (
    field: keyof ModelFormData,
    value: string | boolean | number | null,
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }))

    // Clear error when user starts typing
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: '' }))
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[95vh] overflow-y-auto ultra-modal-scroll p-6">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-xl pb-4">
            {isEditing ? (
              <>
                <span className="text-2xl">📝</span>
                Edit Model:{' '}
                <Badge variant="secondary" className="px-3 py-1">
                  {model?.display_name}
                </Badge>
              </>
            ) : (
              <>
                <span className="text-2xl">➕</span>
                Add New Model
              </>
            )}
          </DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6 p-1">
          {/* Basic Information */}
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="filename">Filename *</Label>
              <Input
                id="filename"
                value={formData.filename}
                onChange={(e) => handleInputChange('filename', e.target.value)}
                placeholder="model_name.safetensors"
                disabled={isEditing} // Don't allow filename changes when editing
                className={errors.filename ? 'border-red-500' : ''}
              />
              {errors.filename && (
                <p className="text-sm text-red-500">{errors.filename}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="display_name">Display Name *</Label>
              <Input
                id="display_name"
                value={formData.display_name}
                onChange={(e) => handleInputChange('display_name', e.target.value)
                }
                placeholder="Human-readable name"
                className={errors.display_name ? 'border-red-500' : ''}
              />
              {errors.display_name && (
                <p className="text-sm text-red-500">{errors.display_name}</p>
              )}
            </div>
          </div>

          {/* Model Configuration */}
          <div className="grid grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="type">Type</Label>
              <Select
                value={formData.type}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange('type', e.target.value)
                }
              >
                {MODEL_TYPES.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="base_model">Base Model</Label>
              <Select
                value={formData.base_model}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange('base_model', e.target.value)
                }
              >
                {BASE_MODELS.map((base) => (
                  <SelectItem key={base.value} value={base.value}>
                    {base.label}
                  </SelectItem>
                ))}
              </Select>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-6">
            <div className="space-y-2">
              <Label htmlFor="file_format">File Format</Label>
              <Select
                value={formData.file_format}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange('file_format', e.target.value)
                }
              >
                {FILE_FORMATS.map((format) => (
                  <SelectItem key={format.value} value={format.value}>
                    {format.label}
                  </SelectItem>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="capability">Capability</Label>
              <Select
                value={formData.capability}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange('capability', e.target.value)
                }
              >
                {CAPABILITIES.map((cap) => (
                  <SelectItem key={cap.value} value={cap.value}>
                    {cap.label}
                  </SelectItem>
                ))}
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="source">Source</Label>
              <Select
                value={formData.source}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => handleInputChange('source', e.target.value)
                }
              >
                {SOURCES.map((source) => (
                  <SelectItem key={source.value} value={source.value}>
                    {source.label}
                  </SelectItem>
                ))}
              </Select>
            </div>
          </div>

          {/* Additional Information */}
          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => handleInputChange('description', e.target.value)
              }
              placeholder="Optional description of the model..."
              rows={3}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="file_size_mb">File Size (MB)</Label>
              <Input
                id="file_size_mb"
                type="number"
                min="0"
                step="0.1"
                value={formData.file_size_mb || ''}
                onChange={(e) => handleInputChange(
                  'file_size_mb',
                  e.target.value ? parseFloat(e.target.value) : null,
                )
                }
                placeholder="e.g. 1024.5"
                className={errors.file_size_mb ? 'border-red-500' : ''}
              />
              {errors.file_size_mb && (
                <p className="text-sm text-red-500">{errors.file_size_mb}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="is_nsfw">NSFW Content</Label>
              <div className="flex items-center gap-2">
                <Switch
                  id="is_nsfw"
                  checked={formData.is_nsfw}
                  onCheckedChange={(checked: boolean) => handleInputChange('is_nsfw', checked)
                  }
                />
                <span className="text-sm text-muted-foreground">
                  Mark if this model generates adult content
                </span>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {submitError && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-600">{submitError}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading && <span className="animate-spin mr-2">⏳</span>}
              {isEditing ? 'Update Model' : 'Add Model'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

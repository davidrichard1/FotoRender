'use client'

import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import {
  Button,
  Input,
  Label,
  Textarea,
  Select,
  SelectItem,
  Switch,
  Badge,
  Card,
  CardContent,
} from '@/components/ui'
import { ApiLora } from '@/lib/api'

// LoRA categories (matching backend enum)
const LORA_CATEGORIES = [
  'GENERAL',
  'DETAIL_ENHANCEMENT',
  'STYLE',
  'CHARACTER',
  'CONCEPT',
  'CLOTHING',
  'POSE',
  'LIGHTING',
  'EXPERIMENTAL',
  'NSFW',
] as const

type LoraCategory = (typeof LORA_CATEGORIES)[number]

// Form data interface
interface LoraFormData {
  filename: string
  displayName: string
  description: string
  category: LoraCategory
  isNsfw: boolean
  isActive: boolean
  defaultScale: number
  minScale: number
  maxScale: number
  recommendedMin: number
  recommendedMax: number
  triggerWords: string[]
  usageTips: string
  author: string
  version: string
  website: string
  tags: string[]
}

interface LoraModalProps {
  isOpen: boolean
  onClose: () => void
  lora?: ApiLora | null // For editing existing LoRA
  onSuccess: (lora: ApiLora) => void
}

export default function LoraModal({
  isOpen,
  onClose,
  lora,
  onSuccess,
}: LoraModalProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentTag, setCurrentTag] = useState('')
  const [currentTriggerWord, setCurrentTriggerWord] = useState('')

  // Form state
  const [formData, setFormData] = useState<LoraFormData>({
    filename: '',
    displayName: '',
    description: '',
    category: 'GENERAL',
    isNsfw: false,
    isActive: true,
    defaultScale: 1.0,
    minScale: -5.0,
    maxScale: 5.0,
    recommendedMin: 0.5,
    recommendedMax: 1.5,
    triggerWords: [],
    usageTips: '',
    author: '',
    version: '',
    website: '',
    tags: [],
  })

  // Reset form when modal opens/closes or when editing different LoRA
  useEffect(() => {
    if (isOpen) {
      if (lora) {
        // Editing existing LoRA
        setFormData({
          filename: lora.filename,
          displayName:
            lora.display_name || lora.filename.replace('.safetensors', ''),
          description: lora.description || '',
          category: (lora.category as LoraCategory) || 'GENERAL',
          isNsfw: lora.is_nsfw || false,
          isActive: true, // Assume active
          defaultScale: lora.default_scale || 1.0,
          minScale: lora.min_scale || -5.0,
          maxScale: lora.max_scale || 5.0,
          recommendedMin: lora.recommended_min || 0.5,
          recommendedMax: lora.recommended_max || 1.5,
          triggerWords: lora.trigger_words || [],
          usageTips: lora.usage_tips || '',
          author: lora.author || '',
          version: lora.version || '',
          website: lora.website || '',
          tags: lora.tags || [],
        })
      } else {
        // Adding new LoRA
        setFormData({
          filename: '',
          displayName: '',
          description: '',
          category: 'GENERAL',
          isNsfw: false,
          isActive: true,
          defaultScale: 1.0,
          minScale: -5.0,
          maxScale: 5.0,
          recommendedMin: 0.5,
          recommendedMax: 1.5,
          triggerWords: [],
          usageTips: '',
          author: '',
          version: '',
          website: '',
          tags: [],
        })
      }
      setError(null)
    }
  }, [isOpen, lora])

  const handleInputChange = (
    field: keyof LoraFormData,
    value: string | number | boolean | string[],
  ) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const addTag = () => {
    if (currentTag.trim() && !formData.tags.includes(currentTag.trim())) {
      setFormData((prev) => ({
        ...prev,
        tags: [...prev.tags, currentTag.trim()],
      }))
      setCurrentTag('')
    }
  }

  const removeTag = (tagToRemove: string) => {
    setFormData((prev) => ({
      ...prev,
      tags: prev.tags.filter((tag) => tag !== tagToRemove),
    }))
  }

  const addTriggerWord = () => {
    if (
      currentTriggerWord.trim()
      && !formData.triggerWords.includes(currentTriggerWord.trim())
    ) {
      setFormData((prev) => ({
        ...prev,
        triggerWords: [...prev.triggerWords, currentTriggerWord.trim()],
      }))
      setCurrentTriggerWord('')
    }
  }

  const removeTriggerWord = (wordToRemove: string) => {
    setFormData((prev) => ({
      ...prev,
      triggerWords: prev.triggerWords.filter((word) => word !== wordToRemove),
    }))
  }

  const validateForm = (): string | null => {
    if (!formData.filename.trim()) {
      return 'Filename is required'
    }
    if (!formData.displayName.trim()) {
      return 'Display name is required'
    }
    if (
      formData.defaultScale < formData.minScale
      || formData.defaultScale > formData.maxScale
    ) {
      return 'Default scale must be between min and max scale'
    }
    if (
      formData.recommendedMin < formData.minScale
      || formData.recommendedMin > formData.maxScale
    ) {
      return 'Recommended min must be within scale range'
    }
    if (
      formData.recommendedMax < formData.minScale
      || formData.recommendedMax > formData.maxScale
    ) {
      return 'Recommended max must be within scale range'
    }
    if (formData.recommendedMin > formData.recommendedMax) {
      return 'Recommended min cannot be greater than recommended max'
    }
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const validationError = validateForm()
    if (validationError) {
      setError(validationError)
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Convert form data to API format
      const apiData: Partial<ApiLora> = {
        filename: formData.filename,
        display_name: formData.displayName,
        description: formData.description,
        category: formData.category,
        is_nsfw: formData.isNsfw,
        default_scale: formData.defaultScale,
        min_scale: formData.minScale,
        max_scale: formData.maxScale,
        recommended_min: formData.recommendedMin,
        recommended_max: formData.recommendedMax,
        trigger_words: formData.triggerWords,
        usage_tips: formData.usageTips,
        author: formData.author,
        version: formData.version,
        website: formData.website,
        tags: formData.tags,
      }

      const apiHost = typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
      let response
      if (lora) {
        // Update existing LoRA
        response = await fetch(`http://${apiHost}/loras/${lora.filename}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(apiData),
        })
      } else {
        // Create new LoRA
        response = await fetch(`http://${apiHost}/loras`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(apiData),
        })
      }

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(
          errorData.error || `Failed to ${lora ? 'update' : 'create'} LoRA`,
        )
      }

      const result = await response.json()
      onSuccess(result)
      onClose()
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'An unexpected error occurred',
      )
    } finally {
      setLoading(false)
    }
  }

  const formatCategoryLabel = (category: string) => category
    .split('_')
    .map((word) => word.charAt(0) + word.slice(1).toLowerCase())
    .join(' ')

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto ultra-modal-scroll">
        <DialogHeader>
          <DialogTitle>{lora ? 'Edit LoRA' : 'Add New LoRA'}</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
              <p className="text-red-600 dark:text-red-400 text-sm">{error}</p>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Basic Information */}
            <Card>
              <CardContent className="p-4 space-y-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Basic Information
                </h3>

                <div className="space-y-2">
                  <Label htmlFor="filename">Filename *</Label>
                  <Input
                    id="filename"
                    value={formData.filename}
                    onChange={(e) => handleInputChange('filename', e.target.value)
                    }
                    placeholder="lora-name.safetensors"
                    disabled={!!lora} // Can't change filename when editing
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="displayName">Display Name *</Label>
                  <Input
                    id="displayName"
                    value={formData.displayName}
                    onChange={(e) => handleInputChange('displayName', e.target.value)
                    }
                    placeholder="Human-readable name"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="description">Description</Label>
                  <Textarea
                    id="description"
                    value={formData.description}
                    onChange={(e) => handleInputChange('description', e.target.value)
                    }
                    placeholder="Describe what this LoRA does..."
                    rows={3}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="category">Category</Label>
                  <Select
                    id="category"
                    value={formData.category}
                    onChange={(e) => handleInputChange(
                      'category',
                        e.target.value as LoraCategory,
                    )
                    }
                  >
                    {LORA_CATEGORIES.map((category) => (
                      <SelectItem key={category} value={category}>
                        {formatCategoryLabel(category)}
                      </SelectItem>
                    ))}
                  </Select>
                </div>

                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="isNsfw"
                      checked={formData.isNsfw}
                      onCheckedChange={(checked) => handleInputChange('isNsfw', checked)
                      }
                    />
                    <Label htmlFor="isNsfw">NSFW Content</Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Switch
                      id="isActive"
                      checked={formData.isActive}
                      onCheckedChange={(checked) => handleInputChange('isActive', checked)
                      }
                    />
                    <Label htmlFor="isActive">Active</Label>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Scale Configuration */}
            <Card>
              <CardContent className="p-4 space-y-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Scale Configuration
                </h3>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="defaultScale">Default Scale</Label>
                    <Input
                      id="defaultScale"
                      type="number"
                      step="0.1"
                      value={formData.defaultScale}
                      onChange={(e) => handleInputChange(
                        'defaultScale',
                        parseFloat(e.target.value) || 0,
                      )
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="minScale">Min Scale</Label>
                    <Input
                      id="minScale"
                      type="number"
                      step="0.1"
                      value={formData.minScale}
                      onChange={(e) => handleInputChange(
                        'minScale',
                        parseFloat(e.target.value) || 0,
                      )
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="maxScale">Max Scale</Label>
                    <Input
                      id="maxScale"
                      type="number"
                      step="0.1"
                      value={formData.maxScale}
                      onChange={(e) => handleInputChange(
                        'maxScale',
                        parseFloat(e.target.value) || 0,
                      )
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="recommendedMin">Rec. Min</Label>
                    <Input
                      id="recommendedMin"
                      type="number"
                      step="0.1"
                      value={formData.recommendedMin}
                      onChange={(e) => handleInputChange(
                        'recommendedMin',
                        parseFloat(e.target.value) || 0,
                      )
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="recommendedMax">Rec. Max</Label>
                    <Input
                      id="recommendedMax"
                      type="number"
                      step="0.1"
                      value={formData.recommendedMax}
                      onChange={(e) => handleInputChange(
                        'recommendedMax',
                        parseFloat(e.target.value) || 0,
                      )
                      }
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="usageTips">Usage Tips</Label>
                  <Textarea
                    id="usageTips"
                    value={formData.usageTips}
                    onChange={(e) => handleInputChange('usageTips', e.target.value)
                    }
                    placeholder="Tips for using this LoRA effectively..."
                    rows={3}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Trigger Words */}
            <Card>
              <CardContent className="p-4 space-y-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Trigger Words
                </h3>

                <div className="flex space-x-2">
                  <Input
                    value={currentTriggerWord}
                    onChange={(e) => setCurrentTriggerWord(e.target.value)}
                    placeholder="Add trigger word..."
                    onKeyPress={(e) => e.key === 'Enter'
                      && (e.preventDefault(), addTriggerWord())
                    }
                  />
                  <Button
                    type="button"
                    onClick={addTriggerWord}
                    variant="outline"
                  >
                    Add
                  </Button>
                </div>

                <div className="flex flex-wrap gap-2">
                  {formData.triggerWords.map((word, index) => (
                    <Badge
                      key={index}
                      variant="secondary"
                      className="cursor-pointer"
                      onClick={() => removeTriggerWord(word)}
                    >
                      {word} ×
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Metadata & Tags */}
            <Card>
              <CardContent className="p-4 space-y-4">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Metadata
                </h3>

                <div className="space-y-2">
                  <Label htmlFor="author">Author</Label>
                  <Input
                    id="author"
                    value={formData.author}
                    onChange={(e) => handleInputChange('author', e.target.value)
                    }
                    placeholder="Creator name"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="version">Version</Label>
                    <Input
                      id="version"
                      value={formData.version}
                      onChange={(e) => handleInputChange('version', e.target.value)
                      }
                      placeholder="v1.0"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="website">Website</Label>
                    <Input
                      id="website"
                      value={formData.website}
                      onChange={(e) => handleInputChange('website', e.target.value)
                      }
                      placeholder="https://..."
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Tags</Label>
                  <div className="flex space-x-2">
                    <Input
                      value={currentTag}
                      onChange={(e) => setCurrentTag(e.target.value)}
                      placeholder="Add tag..."
                      onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())
                      }
                    />
                    <Button type="button" onClick={addTag} variant="outline">
                      Add
                    </Button>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {formData.tags.map((tag, index) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="cursor-pointer"
                        onClick={() => removeTag(tag)}
                      >
                        {tag} ×
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={loading}>
              {(() => {
                if (loading) return 'Saving...'
                if (lora) return 'Update LoRA'
                return 'Add LoRA'
              })()}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

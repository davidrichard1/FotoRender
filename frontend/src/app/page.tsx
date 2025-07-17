'use client'

import { useState, useCallback, useEffect, Suspense } from 'react'
import NextImage from 'next/image'
import { ImageViewer } from '@/components/image'
import { CachedImage } from '@/components/image/CachedImage'
import { Label } from '@/components/ui/label'
import { Spinner } from '@/components/ui/spinner'
import { Accordion } from '@/components/ui/accordion'
import { JobProgress } from '@/components/ui/JobProgress'
import { useJobStatus } from '@/hooks/useJobStatus'
import { useScrollSync } from '@/hooks/useScrollSync'
import { generateImage, cancelJob, GenerationRequest, savePrompt } from '@/lib/api'
import { createDataPromises, refreshData } from '@/lib/data-promises'
import {
  DataErrorBoundary,
  LoadingStates
} from '@/components/ui/DataErrorBoundary'
import { ModelSelector } from '@/components/ModelSelector'
import { LoraSelector } from '@/components/LoraSelector'
import {
  UpscalerSelector,
  EmbeddingSelector,
  VaeSelector
} from '@/components/AssetSelectors'
import { useToast } from '@/components/ui/toast'

interface GeneratedImage {
  id: string
  url: string
  title: string
  description?: string
  originalUrl: string
  platform: string
  isPrivate: boolean
  isFavorite: boolean
  userId: string
  createdAt: string
  updatedAt: string
  tags: { name: string; color: string }[]
  isUpscaled?: boolean
  originalImageUrl?: string
  parameters: {
    prompt: string
    negative_prompt: string
    width: number
    height: number
    steps: number
    guidance_scale: number
    sampler: string
    model: string
    seed: number
    clip_skip: number | null
  }
}

interface GenerationParams {
  prompt: string
  negative_prompt: string
  width: number
  height: number
  num_inference_steps: number
  guidance_scale: number
  seed: number
  sampler: string
  model: string
  clip_skip: number | null
}

interface SelectedLora {
  filename: string
  scale: number
}

export default function HomePage() {
  const { showSuccess, showError, ToastContainer } = useToast()
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedImage, setGeneratedImage] = useState<GeneratedImage | null>(
    null
  )
  const [error, setError] = useState<string | null>(null)
  const [showImageViewer, setShowImageViewer] = useState(false)

  // ===== NEW QUEUE-BASED GENERATION STATE =====
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [lastGenerationParams, setLastGenerationParams] =
    useState<GenerationRequest | null>(null)

  // Real-time job status tracking with WebSocket + polling
  // ===== MODERNIZED STATE MANAGEMENT =====
  const [currentModel, setCurrentModel] = useState<string>('')

  const {
    jobStatus,
    isLoading: jobIsLoading,
    error: jobError
  } = useJobStatus(currentJobId, {
    onComplete: (completedJob) => {
      if (completedJob.result_url) {
        const generatedImageData: GeneratedImage = {
          id: `generated-${Date.now()}`,
          url: completedJob.result_url,
          originalUrl: completedJob.result_url,
          title:
            `Generated: ${lastGenerationParams?.prompt?.substring(0, 50)}...` ||
            'Generated Image',
          description: `Generated with queue system (Job: ${completedJob.job_id})`,
          platform: 'AI Generated (Queue)',
          isPrivate: false,
          isFavorite: false,
          userId: 'current-user',
          createdAt: completedJob.created_at || new Date().toISOString(),
          updatedAt: completedJob.completed_at || new Date().toISOString(),
          isUpscaled: false,
          originalImageUrl: completedJob.result_url,
          tags: [
            { name: 'AI Generated', color: '#f59e0b' },
            { name: 'Queue System', color: '#10b981' }
          ],
          parameters: {
            prompt: lastGenerationParams?.prompt || '',
            negative_prompt: lastGenerationParams?.negative_prompt || '',
            width: lastGenerationParams?.width || 1024,
            height: lastGenerationParams?.height || 1024,
            steps: lastGenerationParams?.num_inference_steps || 30,
            guidance_scale: lastGenerationParams?.guidance_scale || 7.5,
            sampler: lastGenerationParams?.sampler || 'DPM++ 2M SDE Karras',
            model: currentModel || '',
            seed: completedJob.seed || -1,
            clip_skip: lastGenerationParams?.clip_skip || null
          }
        }
        setGeneratedImage(generatedImageData)
        setCurrentJobId(null) // Clear job tracking
      }
    },
    onError: (errorMessage) => {
      setError(errorMessage)
      setIsGenerating(false)
    }
  })
  const [selectedLoras, setSelectedLoras] = useState<SelectedLora[]>([])
  const [selectedUpscaler, setSelectedUpscaler] = useState<string>('')
  const [customUpscaleScale, setCustomUpscaleScale] = useState<number | null>(
    null
  )
  const [currentVae, setCurrentVae] = useState<string>('default')
  const [isUpscaling, setIsUpscaling] = useState(false)
  const [isSavingPrompt, setIsSavingPrompt] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('')
  const [isModelSwitching, setIsModelSwitching] = useState(false)

  // Generation parameters
  const [params, setParams] = useState<GenerationParams>({
    prompt:
      'A beautiful landscape with mountains and a lake at sunset, photorealistic, highly detailed',
    negative_prompt: 'blurry, low quality, distorted, bad anatomy, watermark',
    width: 1024,
    height: 1024,
    num_inference_steps: 30,
    guidance_scale: 7.5,
    seed: -1,
    sampler: 'DPM++ 2M SDE Karras',
    model: '',
    clip_skip: null
  })

  // Accordion state for mobile-friendly collapsible sections
  const [accordionState, setAccordionState] = useState({
    modelAssets: false,
    generationSettings: false
  })

  const toggleAccordion = (section: keyof typeof accordionState) => {
    setAccordionState((prev) => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  // ===== CLIENT-SIDE MOUNTING (React recommended pattern) =====
  const [dataPromises, setDataPromises] = useState<ReturnType<
    typeof createDataPromises
  > | null>(null)

  useEffect(() => {
    setDataPromises(createDataPromises())
  }, [])

  // Refresh data when model changes
  const refreshDataForModel = useCallback((modelName: string) => {
    const newPromises = refreshData('loras', modelName)
    setDataPromises((prev) =>
      prev
        ? {
            ...prev,
            lorasPromise: newPromises.lorasPromise
          }
        : newPromises
    )
  }, [])

  const samplers = [
    'DPM++ 2M SDE Karras',
    'DPM++ 2SA Karras',
    'DPM++ 2M',
    'Euler a',
    'Euler',
    'DDIM',
    'LMS'
  ]

  const dimensionOptions = [
    { label: '1024×1024 (Square)', width: 1024, height: 1024 },
    { label: '1152×896 (Landscape)', width: 1152, height: 896 },
    { label: '896×1152 (Portrait)', width: 896, height: 1152 },
    { label: '1216×832 (Wide Landscape)', width: 1216, height: 832 },
    { label: '832×1216 (Tall Portrait)', width: 832, height: 1216 },
    { label: '1344×768 (Ultra Wide)', width: 1344, height: 768 },
    { label: '768×1344 (Ultra Tall)', width: 768, height: 1344 },
    { label: '1536×640 (Cinematic)', width: 1536, height: 640 },
    { label: '640×1536 (Banner)', width: 640, height: 1536 }
  ]

  // ===== API FUNCTIONS =====
  const switchModel = useCallback(
    async (modelName: string) => {
      setIsModelSwitching(true)

      // Collapse sections when switching models
      setAccordionState({
        modelAssets: false,
        generationSettings: false
      })

      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const response = await fetch(`http://${apiHost}/switch-model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_name: modelName })
        })
        if (!response.ok) {
          throw new Error('Failed to switch model')
        }
        const data = await response.json()
        setCurrentModel(data.current_model)
        setParams((prev) => ({ ...prev, model: modelName }))
        refreshDataForModel(modelName)
      } finally {
        setIsModelSwitching(false)
      }
    },
    [refreshDataForModel]
  )

  const loadVae = useCallback(async (vaeFilename: string) => {
    const apiHost =
      typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
    const response = await fetch(`http://${apiHost}/load-vae`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vae_filename: vaeFilename })
    })
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'Failed to load VAE')
    }
    setCurrentVae(vaeFilename)
  }, [])

  const resetVae = useCallback(async () => {
    const apiHost =
      typeof window !== 'undefined'
        ? `${window.location.hostname}:8000`
        : 'localhost:8000'
    const response = await fetch(`http://${apiHost}/reset-vae`, {
      method: 'POST'
    })
    if (!response.ok) {
      throw new Error('Failed to reset VAE')
    }
    setCurrentVae('default')
  }, [])

  // ===== LORA MANAGEMENT =====
  const addLora = useCallback((filename: string, customScale?: number) => {
    setSelectedLoras((prev) => {
      if (prev.find((lora) => lora.filename === filename)) {
        return prev
      }
      const scale = customScale !== undefined ? customScale : 1.0
      return [...prev, { filename, scale }]
    })
  }, [])

  const removeLora = useCallback((filename: string) => {
    setSelectedLoras((prev) =>
      prev.filter((lora) => lora.filename !== filename)
    )
  }, [])

  const updateLoraScale = useCallback((filename: string, scale: number) => {
    setSelectedLoras((prev) =>
      prev.map((lora) =>
        lora.filename === filename ? { ...lora, scale } : lora
      )
    )
  }, [])

  const clearAllLoras = useCallback(() => {
    setSelectedLoras([])
  }, [])

  // ===== EMBEDDING MANAGEMENT =====
  const addEmbeddingToPrompt = useCallback(
    (triggerWord: string, isNegative: boolean = false) => {
      if (isNegative) {
        setParams((prev) => ({
          ...prev,
          negative_prompt: prev.negative_prompt
            ? `${triggerWord}, ${prev.negative_prompt}`
            : triggerWord
        }))
      } else {
        setParams((prev) => ({
          ...prev,
          prompt: prev.prompt ? `${prev.prompt}, ${triggerWord}` : triggerWord
        }))
      }
    },
    []
  )

  // ===== UPSCALING =====
  const upscaleImage = useCallback(
    async (imageBase64: string) => {
      if (!selectedUpscaler) {
        throw new Error('Please select an upscaler first')
      }

      if (generatedImage?.isUpscaled) {
        throw new Error('This image has already been upscaled!')
      }

      setIsUpscaling(true)
      setLoadingMessage('Upscaling your image...')

      try {
        const apiHost =
          typeof window !== 'undefined'
            ? `${window.location.hostname}:8000`
            : 'localhost:8000'
        const base64Data = imageBase64.includes(',')
          ? imageBase64.split(',')[1]
          : imageBase64
        const scaleToUse = customUpscaleScale || 4

        const response = await fetch(`http://${apiHost}/upscale`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: base64Data,
            upscaler_filename: selectedUpscaler,
            scale_factor: scaleToUse
          })
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Upscaling failed')
        }

        const result = await response.json()
        const upscaledImageUrl = `data:image/png;base64,${result.upscaled_image}`

        if (generatedImage) {
          setGeneratedImage({
            ...generatedImage,
            url: upscaledImageUrl,
            originalUrl: upscaledImageUrl,
            title: `Upscaled: ${generatedImage.title}`,
            description: `${generatedImage.description} (Upscaled ${result.scale_factor_used}x)`,
            isUpscaled: true,
            originalImageUrl:
              generatedImage.originalImageUrl || generatedImage.url,
            tags: [
              ...generatedImage.tags,
              {
                name: `Upscaled ${result.scale_factor_used}x`,
                color: '#f59e0b'
              },
              {
                name: result.upscaler_used.type.toUpperCase(),
                color: '#8b5cf6'
              }
            ],
            parameters: {
              ...generatedImage.parameters,
              width: result.upscaled_size.width,
              height: result.upscaled_size.height
            }
          })
        }
      } finally {
        setIsUpscaling(false)
        setLoadingMessage('')
      }
    },
    [selectedUpscaler, customUpscaleScale, generatedImage]
  )

  const resetToOriginal = useCallback(() => {
    if (generatedImage?.originalImageUrl && generatedImage.isUpscaled) {
      setGeneratedImage({
        ...generatedImage,
        url: generatedImage.originalImageUrl,
        originalUrl: generatedImage.originalImageUrl,
        title: generatedImage.title.replace('Upscaled: ', ''),
        description:
          generatedImage.description?.replace(/ \(Upscaled.*?\)/, '') || '',
        isUpscaled: false,
        tags: generatedImage.tags.filter(
          (tag) =>
            !tag.name.includes('Upscaled') &&
            !['REAL-ESRGAN', 'ESRGAN', 'ANIME', 'PHOTO', 'GENERAL'].includes(
              tag.name
            )
        )
      })
    }
  }, [generatedImage])

  // ===== GENERATION FUNCTIONS =====
  const handleGenerate = useCallback(async () => {
    setIsGenerating(true)
    setGeneratedImage(null)
    setError('')
    setLoadingMessage('Submitting to queue...')

    try {
      const requestBody: GenerationRequest = {
        prompt: params.prompt,
        negative_prompt: params.negative_prompt,
        width: params.width,
        height: params.height,
        num_inference_steps: params.num_inference_steps,
        guidance_scale: params.guidance_scale,
        seed: params.seed,
        sampler: params.sampler,
        loras: selectedLoras.map((lora) => ({
          filename: lora.filename,
          scale: lora.scale
        })),
        ...(params.clip_skip !== null && { clip_skip: params.clip_skip }),
        model_name: currentModel || params.model || '' // Always include model_name
      }

      setLastGenerationParams(requestBody)
      const response = await generateImage(requestBody)

      if (response.error || !response.data) {
        throw new Error(response.error || 'Failed to queue generation')
      }

      const jobData = response.data
      if (jobData && jobData.job_id) {
        setCurrentJobId(jobData.job_id)
        setLoadingMessage(
          `Queued successfully! Job ID: ${jobData.job_id.slice(0, 8)}...`
        )
      } else if (jobData && jobData.image) {
        // Monolithic mode: display the image directly
        setGeneratedImage({
          id: `generated-${Date.now()}`,
          url: `data:image/png;base64,${jobData.image}`,
          originalUrl: `data:image/png;base64,${jobData.image}`,
          title: 'Generated Image',
          description: 'Generated in monolithic mode',
          platform: 'AI Generated (Monolith)',
          isPrivate: false,
          isFavorite: false,
          userId: 'current-user',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          isUpscaled: false,
          originalImageUrl: `data:image/png;base64,${jobData.image}`,
          tags: [
            { name: 'AI Generated', color: '#f59e0b' },
            { name: 'Monolith', color: '#10b981' }
          ],
          parameters: jobData.parameters || {}
        })
        setLoadingMessage('Image generated successfully!')
      } else {
        setCurrentJobId(null)
        setLoadingMessage('Image generation started (monolithic mode or unknown job id).')
      }
      setIsGenerating(false)
    } catch (err) {
      console.error('❌ Failed to queue generation:', err)
      setError(
        err instanceof Error ? err.message : 'Failed to queue generation'
      )
      setIsGenerating(false)
      setLoadingMessage('')
    }
  }, [params, selectedLoras, currentModel, params.model])

  const handleCancelJob = useCallback(async () => {
    if (!currentJobId) return
    try {
      const response = await cancelJob(currentJobId)
      if (response.data) {
        setCurrentJobId(null)
        setIsGenerating(false)
      }
    } catch (err) {
      console.error('❌ Failed to cancel job:', err)
    }
  }, [currentJobId])

  const handleRetryGeneration = useCallback(() => {
    if (lastGenerationParams) {
      handleGenerate()
    }
  }, [lastGenerationParams, handleGenerate])

  const handleSavePrompt = async () => {
    if (!generatedImage) return
    setIsSavingPrompt(true)
    try {
      const imagePayload = generatedImage.url.startsWith('data:image')
        ? { image_base64: generatedImage.url }
        : { image_url: generatedImage.url }

      const payload = {
        title: generatedImage.title || 'Untitled',
        prompt: generatedImage.parameters.prompt,
        negative_prompt: generatedImage.parameters.negative_prompt,
        description: generatedImage.description || '',
        width: generatedImage.parameters.width,
        height: generatedImage.parameters.height,
        steps: generatedImage.parameters.steps,
        guidance_scale: generatedImage.parameters.guidance_scale,
        seed: generatedImage.parameters.seed,
        sampler: generatedImage.parameters.sampler,
        clip_skip: generatedImage.parameters.clip_skip,
        loras_used: selectedLoras.map((l) => ({ filename: l.filename, scale: l.scale })),
        model_name: generatedImage.parameters.model,
        ...imagePayload,
        is_public: false
      }

      const { error } = await savePrompt(payload as any)
      if (error) {
        showError(`Failed to save prompt: ${error}`)
      } else {
        showSuccess('Prompt saved successfully!')
      }
    } catch (err: any) {
      showError(`Error saving prompt: ${err.message || err}`)
    } finally {
      setIsSavingPrompt(false)
    }
  }

  // Initialize optimized scroll synchronization
  useScrollSync()

  return (
    <DataErrorBoundary fallbackTitle="Failed to load application">
      <div className="min-h-screen lg:h-screen bg-[#0D1117] flex flex-col lg:overflow-hidden w-full">
        {/* Mobile-responsive header */}
        <div className="bg-[#161B22] border-b border-[#30363D] px-4 sm:px-6 py-3 flex-shrink-0 overflow-visible">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 sm:gap-6 min-w-0">
              <h1 className="text-base sm:text-lg font-medium text-white flex-shrink-0">
                Foto-Render
              </h1>
              <div className="hidden sm:block text-xs text-gray-500">
                AI Image Generation
              </div>
            </div>

            {/* Model Selector with Suspense */}
            <div className="flex items-center gap-2 sm:gap-3">
              <DataErrorBoundary fallbackTitle="Failed to load models">
                <Suspense fallback={<LoadingStates.Models />}>
                  {dataPromises ? (
                    <ModelSelector
                      modelsPromise={dataPromises.modelsPromise}
                      currentModel={currentModel}
                      onModelChange={setCurrentModel}
                      onModelSwitch={switchModel}
                    />
                  ) : (
                    <LoadingStates.Models />
                  )}
                </Suspense>
              </DataErrorBoundary>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="p-4 sm:p-6 lg:h-[calc(100vh-116px)] flex-1 lg:flex-none overflow-y-auto lg:overflow-hidden w-full max-w-full">
          <div className="flex flex-col lg:flex-row gap-4 lg:gap-6 lg:h-full">
            {/* Generation Panel */}
            <div className="w-full lg:w-[clamp(320px,_35vw,_480px)] bg-[#161B22] border border-[#30363D] rounded-lg flex flex-col flex-shrink-0 generation-panel lg:h-full">
              <div className="flex items-center justify-between p-4 md:p-6 pb-4 flex-shrink-0">
                <h2 className="text-lg font-medium text-white">
                  Generation Parameters
                </h2>
              </div>

              <div className="flex-1 px-4 md:px-6 pb-4 md:pb-6 lg:overflow-y-auto space-y-6">
                {/* Prompt */}
                <div>
                  <Label className="text-white font-medium mb-2 block">
                    Prompt
                  </Label>
                  <textarea
                    value={params.prompt}
                    onChange={(e) =>
                      setParams((prev) => ({ ...prev, prompt: e.target.value }))
                    }
                    className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors resize-none"
                    style={{
                      minHeight: '100px',
                      maxHeight: '200px'
                    }}
                    rows={4}
                    placeholder="Describe what you want to generate..."
                  />
                </div>

                {/* Negative Prompt */}
                <div>
                  <Label className="text-white font-medium mb-2 block">
                    Negative Prompt
                  </Label>
                  <textarea
                    value={params.negative_prompt}
                    onChange={(e) =>
                      setParams((prev) => ({
                        ...prev,
                        negative_prompt: e.target.value
                      }))
                    }
                    className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors resize-none"
                    style={{
                      minHeight: '60px',
                      maxHeight: '160px'
                    }}
                    rows={2}
                    placeholder="What to avoid..."
                  />
                </div>

                {/* Model Assets with Suspense */}
                <Accordion
                  title="Model Assets"
                  subtitle={`${selectedLoras.length} LoRAs • VAE: ${
                    currentVae === 'default' ? 'Default' : currentVae
                  }`}
                  isOpen={accordionState.modelAssets}
                  onToggle={() => toggleAccordion('modelAssets')}
                >
                  {/* LoRA Management */}
                  <DataErrorBoundary fallbackTitle="Failed to load LoRAs">
                    <Suspense fallback={<LoadingStates.Loras />}>
                      {dataPromises ? (
                        <LoraSelector
                          lorasPromise={dataPromises.lorasPromise}
                          selectedLoras={selectedLoras}
                          onLoraAdd={addLora}
                          onLoraRemove={removeLora}
                          onLoraScaleUpdate={updateLoraScale}
                          onClearAll={clearAllLoras}
                        />
                      ) : (
                        <LoadingStates.Loras />
                      )}
                    </Suspense>
                  </DataErrorBoundary>

                  {/* Embeddings Section */}
                  <DataErrorBoundary fallbackTitle="Failed to load embeddings">
                    <Suspense fallback={<LoadingStates.Embeddings />}>
                      {dataPromises ? (
                        <EmbeddingSelector
                          embeddingsPromise={dataPromises.embeddingsPromise}
                          onAddToPrompt={addEmbeddingToPrompt}
                        />
                      ) : (
                        <LoadingStates.Embeddings />
                      )}
                    </Suspense>
                  </DataErrorBoundary>

                  {/* VAE Section */}
                  <DataErrorBoundary fallbackTitle="Failed to load VAEs">
                    <Suspense fallback={<LoadingStates.Vaes />}>
                      {dataPromises ? (
                        <VaeSelector
                          vaesPromise={dataPromises.vaesPromise}
                          currentVae={currentVae}
                          onVaeLoad={loadVae}
                          onVaeReset={resetVae}
                        />
                      ) : (
                        <LoadingStates.Vaes />
                      )}
                    </Suspense>
                  </DataErrorBoundary>
                </Accordion>

                {/* Generation Settings */}
                <Accordion
                  title="Generation Settings"
                  subtitle={`${params.width}×${params.height} • ${params.num_inference_steps} steps • ${params.guidance_scale} guidance • ${params.sampler}`}
                  isOpen={accordionState.generationSettings}
                  onToggle={() => toggleAccordion('generationSettings')}
                >
                  {/* Dimensions */}
                  <div className="mb-4">
                    <Label className="text-white font-medium mb-2 block">
                      Dimensions
                    </Label>
                    <select
                      value={`${params.width}x${params.height}`}
                      onChange={(e) => {
                        const [width, height] = e.target.value
                          .split('x')
                          .map(Number)
                        setParams((prev) => ({ ...prev, width, height }))
                      }}
                      className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                    >
                      {dimensionOptions.map((option) => (
                        <option
                          key={`dimension-${option.width}x${option.height}`}
                          value={`${option.width}x${option.height}`}
                          className="bg-[#0D1117] text-white"
                        >
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Parameters Grid */}
                  <div className="mb-4">
                    <Label className="text-white font-medium mb-2 block">
                      Parameters
                    </Label>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 param-grid">
                        <div>
                          <Label className="text-white font-medium mb-2 block text-sm">
                            Steps
                          </Label>
                          <input
                            type="number"
                            value={params.num_inference_steps}
                            onChange={(e) =>
                              setParams((prev) => ({
                                ...prev,
                                num_inference_steps: parseInt(
                                  e.target.value,
                                  10
                                )
                              }))
                            }
                            className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                            min="1"
                            max="100"
                          />
                        </div>
                        <div>
                          <Label className="text-white font-medium mb-2 block text-sm">
                            Guidance Scale
                          </Label>
                          <input
                            type="number"
                            value={params.guidance_scale}
                            onChange={(e) =>
                              setParams((prev) => ({
                                ...prev,
                                guidance_scale: parseFloat(e.target.value)
                              }))
                            }
                            className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                            min="1"
                            max="20"
                            step="0.1"
                          />
                        </div>
                      </div>

                      {/* CLIP Skip and Sampler Grid */}
                      <div className="grid grid-cols-2 gap-4 param-grid">
                        <div>
                          <Label className="text-white font-medium mb-2 block text-sm">
                            CLIP Skip
                          </Label>
                          <input
                            type="number"
                            value={params.clip_skip || ''}
                            onChange={(e) =>
                              setParams((prev) => ({
                                ...prev,
                                clip_skip: e.target.value
                                  ? parseInt(e.target.value, 10)
                                  : null
                              }))
                            }
                            className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                            min="1"
                            max="12"
                            placeholder="Auto"
                          />
                          <p className="text-xs text-gray-400 mt-1">
                            {params.clip_skip !== null
                              ? `Set: ${params.clip_skip} - Higher = more creative`
                              : "Using model's default behavior"}
                          </p>
                        </div>
                        <div>
                          <Label className="text-white font-medium mb-2 block text-sm">
                            Sampler
                          </Label>
                          <select
                            value={params.sampler}
                            onChange={(e) =>
                              setParams((prev) => ({
                                ...prev,
                                sampler: e.target.value
                              }))
                            }
                            className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                          >
                            {samplers.map((sampler) => (
                              <option
                                key={`sampler-${sampler}`}
                                value={sampler}
                              >
                                {sampler}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {/* Seed */}
                      <div>
                        <Label className="text-white font-medium mb-2 block text-sm">
                          Seed (-1 for random)
                        </Label>
                        <input
                          type="number"
                          value={params.seed}
                          onChange={(e) =>
                            setParams((prev) => ({
                              ...prev,
                              seed: parseInt(e.target.value, 10)
                            }))
                          }
                          className="w-full px-3 py-2 bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                          min="-1"
                        />
                      </div>
                    </div>
                  </div>
                </Accordion>

                {/* Error Display */}
                {error && (
                  <div className="p-4 bg-red-500/20 border border-red-500/30 rounded-xl text-red-400 text-sm">
                    {error}
                  </div>
                )}
              </div>

              {/* Generate Button - Fixed at bottom */}
              <div className="p-4 md:p-6 pt-4 mt-auto border-t border-[#30363D] bg-[#161B22] flex-shrink-0">
                {!currentModel && (
                  <div className="p-2 bg-yellow-500/20 border border-yellow-500/30 rounded text-yellow-700 text-xs mb-2">
                    Please select a model before generating.
                  </div>
                )}
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating || jobIsLoading || !!currentJobId || !currentModel}
                  className="w-full py-2.5 px-4 bg-[#238636] hover:bg-[#2ea043] disabled:bg-[#30363D] disabled:text-gray-500 text-white text-sm font-medium rounded-md transition-colors flex items-center justify-center space-x-2 border border-[#238636] hover:border-[#2ea043] disabled:border-[#30363D]"
                >
                  {(() => {
                    if (isGenerating) {
                      return (
                        <>
                          <Spinner className="w-4 h-4" />
                          <span>Submitting to Queue...</span>
                        </>
                      )
                    }
                    if (currentJobId) {
                      return (
                        <>
                          <Spinner className="w-4 h-4" />
                          <span>Job in Progress...</span>
                        </>
                      )
                    }
                    return <span>🚀 Generate Image (Instant Queue)</span>
                  })()}
                </button>

                {/* Real-time Job Progress */}
                {(currentJobId || jobIsLoading || jobError) && (
                  <div className="mt-4">
                    <JobProgress
                      jobStatus={jobStatus}
                      isLoading={jobIsLoading}
                      error={jobError}
                      onCancel={handleCancelJob}
                      onRetry={handleRetryGeneration}
                      className="bg-[#0D1117] border-[#30363D]"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Result Panel */}
            <div className="flex-1 bg-[#161B22] border border-[#30363D] rounded-lg flex flex-col min-w-0 lg:h-full">
              <div className="flex items-center justify-between p-6 pb-4 flex-shrink-0">
                <h2 className="text-lg font-medium text-white">
                  Generated Image
                </h2>
                {generatedImage && (
                  <button
                    onClick={handleSavePrompt}
                    disabled={isSavingPrompt}
                    className="px-3 py-1 bg-[#238636] hover:bg-[#2ea043] disabled:bg-[#30363D] disabled:text-gray-500 text-white rounded text-sm font-medium border border-[#238636] hover:border-[#2ea043] disabled:border-[#30363D] transition-colors flex items-center space-x-1"
                  >
                    {isSavingPrompt ? (
                      <>
                        <Spinner className="w-4 h-4" />
                        <span>Saving...</span>
                      </>
                    ) : (
                      <>
                        <span>💾</span>
                        <span>Save Prompt</span>
                      </>
                    )}
                  </button>
                )}
              </div>

              <div className="flex-1 px-6 pb-6 lg:overflow-y-auto">
                <div className="space-y-6">
                  {generatedImage ? (
                    <div className="space-y-6">
                      <div className="relative">
                        <div
                          className="border border-[#30363D] rounded overflow-hidden relative bg-black flex items-center justify-center"
                          style={{ height: 'min(75vh, calc(100vh - 250px))' }}
                        >
                          <NextImage
                            src={generatedImage.url}
                            alt={generatedImage.title}
                            width={generatedImage.parameters.width || 1024}
                            height={generatedImage.parameters.height || 1024}
                            className="w-full h-full object-contain cursor-pointer"
                            loading="lazy"
                            onClick={() => setShowImageViewer(true)}
                          />
                        </div>
                        {generatedImage.isUpscaled && (
                          <div className="absolute top-2 left-2 bg-green-600/20 px-2 py-1 rounded text-green-400 text-sm border border-green-600/40">
                            Upscaled
                          </div>
                        )}
                      </div>

                      {/* Upscaling Controls with Suspense */}
                      <div className="bg-[#161B22] rounded border border-[#30363D] p-4">
                        <div className="flex items-center justify-between mb-4">
                          <Label className="text-white font-semibold text-base flex items-center gap-2">
                            AI Upscaling
                          </Label>
                          {generatedImage.isUpscaled && (
                            <button
                              onClick={resetToOriginal}
                              className="px-3 py-1 bg-[#21262D] hover:bg-[#30363D] text-white rounded text-sm font-medium transition-colors border border-[#30363D]"
                            >
                              Reset to Original
                            </button>
                          )}
                        </div>

                        {!generatedImage.isUpscaled ? (
                          <DataErrorBoundary fallbackTitle="Failed to load upscalers">
                            <Suspense fallback={<LoadingStates.Upscalers />}>
                              {dataPromises ? (
                                <UpscalerSelector
                                  upscalersPromise={
                                    dataPromises.upscalersPromise
                                  }
                                  selectedUpscaler={selectedUpscaler}
                                  customUpscaleScale={customUpscaleScale}
                                  onUpscalerChange={setSelectedUpscaler}
                                  onCustomScaleChange={setCustomUpscaleScale}
                                />
                              ) : (
                                <LoadingStates.Upscalers />
                              )}
                              <button
                                onClick={() => upscaleImage(generatedImage.url)}
                                disabled={!selectedUpscaler || isUpscaling}
                                className="w-full py-2.5 px-4 bg-[#238636] hover:bg-[#2ea043] disabled:bg-[#30363D] disabled:text-gray-500 text-white text-sm font-medium rounded-md transition-colors flex items-center justify-center space-x-2 border border-[#238636] hover:border-[#2ea043] disabled:border-[#30363D] mt-4"
                              >
                                {isUpscaling ? (
                                  <>
                                    <Spinner className="w-5 h-5" />
                                    <span>Upscaling...</span>
                                  </>
                                ) : (
                                  <span>Upscale Image</span>
                                )}
                              </button>
                            </Suspense>
                          </DataErrorBoundary>
                        ) : (
                          <div className="text-center p-4">
                            <div className="text-green-400 mb-2">
                              Image Successfully Upscaled!
                            </div>
                            <p className="text-gray-400 text-sm">
                              Use &quot;Reset to Original&quot; to get the
                              unupscaled version, then upscale again if needed
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Image Info */}
                      <div className="space-y-4">
                        <div className="flex flex-wrap gap-2">
                          {generatedImage.tags.map((tag, index) => (
                            <span
                              key={`tag-${tag.name}-${index}`}
                              className="px-2 py-1 bg-[#21262D] text-gray-300 rounded text-xs border border-[#30363D]"
                            >
                              {tag.name}
                            </span>
                          ))}
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                          <div className="p-4 bg-[#161B22] rounded border border-[#30363D]">
                            <h3 className="text-sm font-medium text-gray-300 mb-3">
                              Generation Settings
                            </h3>
                            <div className="text-sm text-white space-y-2">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Model:</span>
                                <span className="font-medium">
                                  {generatedImage.parameters.model}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">
                                  Dimensions:
                                </span>
                                <span className="font-medium">
                                  {generatedImage.parameters.width}×
                                  {generatedImage.parameters.height}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Steps:</span>
                                <span className="font-medium">
                                  {generatedImage.parameters.steps}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Guidance:</span>
                                <span className="font-medium">
                                  {generatedImage.parameters.guidance_scale}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Sampler:</span>
                                <span className="font-medium">
                                  {generatedImage.parameters.sampler}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">
                                  CLIP Skip:
                                </span>
                                <span className="font-medium">
                                  {generatedImage.parameters.clip_skip ??
                                    'Default'}
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="p-4 bg-[#161B22] rounded border border-[#30363D]">
                            <h3 className="text-sm font-medium text-gray-300 mb-3">
                              Generation Info
                            </h3>
                            <div className="text-sm text-white space-y-2">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Seed:</span>
                                <span className="font-mono text-xs bg-[#21262D] px-2 py-1 rounded border border-[#30363D]">
                                  {generatedImage.parameters.seed}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Created:</span>
                                <span className="font-medium">
                                  {new Date(
                                    generatedImage.createdAt
                                  ).toLocaleTimeString()}
                                </span>
                              </div>
                              <div className="flex justify-between items-center">
                                <span className="text-gray-400">Status:</span>
                                <span className="flex items-center gap-1 text-green-400">
                                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                  Complete
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="aspect-[16/9] bg-[#161B22] rounded border-2 border-dashed border-[#30363D] flex items-center justify-center">
                      <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 bg-[#21262D] rounded-lg flex items-center justify-center text-2xl">
                          🎨
                        </div>
                        <p className="text-white text-lg font-medium">
                          Your generated image will appear here
                        </p>
                        <p className="text-gray-400 text-sm mt-2">
                          Configure your parameters and click Generate
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Image Viewer Modal */}
        {showImageViewer && generatedImage && (
          <ImageViewer
            image={generatedImage}
            onClose={() => setShowImageViewer(false)}
            onToggleFavorite={() => {}}
            onEdit={() => {}}
          />
        )}

        {/* Loading Overlay */}
        {(isUpscaling || isModelSwitching) && (
          <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center">
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6 shadow-xl text-center max-w-sm mx-4">
              <div className="w-8 h-8 mx-auto mb-4 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin"></div>
              <h3 className="text-sm font-medium text-white mb-2">
                {isModelSwitching
                  ? 'Switching Model...'
                  : loadingMessage || 'Processing...'}
              </h3>
              <p className="text-xs text-gray-400">
                {isModelSwitching
                  ? 'Loading new model and refreshing LoRAs...'
                  : isUpscaling && 'Upscaling image...'}
              </p>
            </div>
          </div>
        )}
      </div>
      <ToastContainer />
    </DataErrorBoundary>
  )
}

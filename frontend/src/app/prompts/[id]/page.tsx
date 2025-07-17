'use client'

import { useState, useEffect, Suspense } from 'react'
import { notFound } from 'next/navigation'
import { getPrompt } from '@/lib/api'
import { ImageViewer } from '@/components/image'
import { CachedImage } from '@/components/image/CachedImage'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { useToast } from '@/components/ui/toast'
import Link from 'next/link'
// Using inline SVG icons to match project conventions
const ArrowLeftIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
  </svg>
)

const DownloadIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
)

const InfoIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const EyeIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
  </svg>
)

const CopyIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
)

interface PromptDetailClientProps {
  promptId: string
}

function PromptDetailClient({ promptId }: PromptDetailClientProps) {
  const [prompt, setPrompt] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showImageViewer, setShowImageViewer] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const { showSuccess, showError } = useToast()

  useEffect(() => {
    async function fetchPromptData() {
      try {
        setLoading(true)
        const { data, error } = await getPrompt(promptId)
        if (error) {
          setError(error)
          return
        }
        setPrompt(data)
      } catch (err: any) {
        setError(err.message || 'Failed to load prompt')
      } finally {
        setLoading(false)
      }
    }

    fetchPromptData()
  }, [promptId])

  const handleCopyPrompt = async () => {
    if (!prompt) return
    try {
      await navigator.clipboard.writeText(prompt.prompt)
      showSuccess('Prompt copied to clipboard!')
    } catch (err) {
      showError('Failed to copy prompt')
    }
  }

  const handleCopyNegativePrompt = async () => {
    if (!prompt?.negative_prompt) return
    try {
      await navigator.clipboard.writeText(prompt.negative_prompt)
      showSuccess('Negative prompt copied to clipboard!')
    } catch (err) {
      showError('Failed to copy negative prompt')
    }
  }

  const handleDownloadImage = () => {
    if (!prompt?.image_url) return
    const link = document.createElement('a')
    link.href = prompt.image_url
    link.download = `${prompt.title || 'prompt'}.png`
    link.click()
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="text-gray-400 mt-4">Loading prompt details...</p>
        </div>
      </div>
    )
  }

  if (error || !prompt) {
    return notFound()
  }

  // Convert to ImageViewer format
  const imageViewerData = {
    id: prompt.id,
    url: prompt.image_url,
    title: prompt.title,
    description: prompt.description,
    originalUrl: prompt.image_url,
    platform: 'foto-render',
    isPrivate: !prompt.is_public,
    isFavorite: true, // Since it's saved in the library
    userId: 'current-user',
    createdAt: prompt.created_at,
    updatedAt: prompt.created_at,
    tags: prompt.tags?.map((tag: string) => ({ name: tag, color: '#00D4AA' })) || [],
    parameters: {
      prompt: prompt.prompt,
      negative_prompt: prompt.negative_prompt || '',
      width: prompt.parameters?.width || 1024,
      height: prompt.parameters?.height || 1024,
      steps: prompt.parameters?.steps || 30,
      guidance_scale: prompt.parameters?.guidance_scale || 7.5,
      sampler: prompt.parameters?.sampler || 'DPM++ 2M SDE Karras',
      model: prompt.model_name || 'Unknown',
      seed: prompt.parameters?.seed || -1,
      clip_skip: prompt.parameters?.clip_skip || null,
    }
  }

  return (
    <div className="min-h-screen bg-[#0D1117]">
      {/* Header */}
      <div className="bg-[#161B22] border-b border-[#30363D] px-4 sm:px-6 py-4">
        <div className="container mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
                         <Link 
               href="/prompts"
               className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
             >
               <ArrowLeftIcon />
               <span>Back to Library</span>
             </Link>
          </div>
          
          <div className="flex items-center gap-3">
                         <Button
               variant="outline"
               size="sm"
               onClick={handleDownloadImage}
               className="flex items-center gap-2"
             >
               <DownloadIcon />
               Download
             </Button>
             <Button
               variant="outline"
               size="sm"
               onClick={() => setShowDetails(!showDetails)}
               className="flex items-center gap-2"
             >
               <InfoIcon />
               {showDetails ? 'Hide' : 'Show'} Details
             </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 sm:px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Image Section */}
          <div className="space-y-4">
                        <Card className="overflow-hidden bg-[#161B22] border-[#30363D]">
              <div className="relative group cursor-pointer" onClick={() => setShowImageViewer(true)}>
                <CachedImage
                  src={prompt.image_url}
                  alt={prompt.title}
                  className="w-full h-auto rounded-lg transition-opacity group-hover:opacity-90"
                  loadingComponent={
                    <div className="flex items-center justify-center w-full aspect-square bg-[#161B22]">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#00D4AA]"></div>
                    </div>
                  }
                />
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors rounded-lg flex items-center justify-center">
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="bg-black/80 rounded-full p-3">
                      <EyeIcon />
                    </div>
                  </div>
                </div>
              </div>
            </Card>

            {/* Image Actions */}
            <div className="flex flex-wrap gap-2">
                             <Button
                 variant="outline"
                 size="sm"
                 onClick={() => setShowImageViewer(true)}
                 className="flex items-center gap-2"
               >
                 <EyeIcon />
                 Full View
               </Button>
               <Button
                 variant="outline"
                 size="sm"
                 onClick={handleDownloadImage}
                 className="flex items-center gap-2"
               >
                 <DownloadIcon />
                 Download
               </Button>
            </div>
          </div>

          {/* Details Section */}
          <div className="space-y-6">
            {/* Title and Description */}
            <div>
              <h1 className="text-2xl font-bold text-white mb-2">{prompt.title}</h1>
              {prompt.description && (
                <p className="text-gray-400">{prompt.description}</p>
              )}
              
              {/* Tags */}
              {prompt.tags?.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {prompt.tags.map((tag: string, index: number) => (
                    <Badge key={index} variant="secondary" className="bg-[#00D4AA]/20 text-[#00D4AA]">
                      {tag}
                    </Badge>
                  ))}
                </div>
              )}
            </div>

            {/* Prompt */}
            <Card className="p-4 bg-[#161B22] border-[#30363D]">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-white">Prompt</h3>
                                 <Button
                   variant="ghost"
                   size="sm"
                   onClick={handleCopyPrompt}
                   className="flex items-center gap-2"
                 >
                   <CopyIcon />
                   Copy
                 </Button>
              </div>
              <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">
                {prompt.prompt}
              </p>
            </Card>

            {/* Negative Prompt */}
            {prompt.negative_prompt && (
              <Card className="p-4 bg-[#161B22] border-[#30363D]">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-white">Negative Prompt</h3>
                                     <Button
                     variant="ghost"
                     size="sm"
                     onClick={handleCopyNegativePrompt}
                     className="flex items-center gap-2"
                   >
                     <CopyIcon />
                     Copy
                   </Button>
                </div>
                <p className="text-gray-400 whitespace-pre-wrap leading-relaxed">
                  {prompt.negative_prompt}
                </p>
              </Card>
            )}

            {/* Parameters (collapsible) */}
            {showDetails && prompt.parameters && (
              <Card className="p-4 bg-[#161B22] border-[#30363D]">
                <h3 className="text-lg font-semibold text-white mb-3">Generation Parameters</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white ml-2">{prompt.model_name || 'Unknown'}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Sampler:</span>
                    <span className="text-white ml-2">{prompt.parameters.sampler}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Steps:</span>
                    <span className="text-white ml-2">{prompt.parameters.steps}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Guidance Scale:</span>
                    <span className="text-white ml-2">{prompt.parameters.guidance_scale}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Dimensions:</span>
                    <span className="text-white ml-2">{prompt.parameters.width}×{prompt.parameters.height}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Seed:</span>
                    <span className="text-white ml-2">{prompt.parameters.seed}</span>
                  </div>
                  {prompt.parameters.clip_skip && (
                    <div>
                      <span className="text-gray-400">CLIP Skip:</span>
                      <span className="text-white ml-2">{prompt.parameters.clip_skip}</span>
                    </div>
                  )}
                </div>

                {/* LoRAs */}
                {prompt.loras_used?.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-white font-medium mb-2">LoRAs Used:</h4>
                    <div className="space-y-1">
                      {prompt.loras_used.map((lora: any, index: number) => (
                        <div key={index} className="text-sm">
                          <span className="text-gray-400">{lora.filename}:</span>
                          <span className="text-white ml-2">{lora.scale}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Image Viewer Modal */}
      {showImageViewer && (
        <ImageViewer
          image={imageViewerData}
          onClose={() => setShowImageViewer(false)}
          onToggleFavorite={() => {}} // No-op since it's already favorited
          onEdit={() => {}} // No-op for now
        />
      )}
    </div>
  )
}

export default function PromptDetailPage({ params }: { params: { id: string } }) {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="text-gray-400 mt-4">Loading prompt details...</p>
        </div>
      </div>
    }>
      <PromptDetailClient promptId={params.id} />
    </Suspense>
  )
} 
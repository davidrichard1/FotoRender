'use client'

import { useState, useEffect, Suspense, useMemo } from 'react'
import { getPrompts } from '@/lib/api'
import { ImageViewer } from '@/components/image'
import { CachedImage } from '@/components/image/CachedImage'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { useToast } from '@/components/ui/toast'

import Link from 'next/link'

// SVG Icons
const EyeIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
  </svg>
)

const HeartIcon = ({ filled = false }: { filled?: boolean }) => (
  <svg className="w-5 h-5" fill={filled ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
  </svg>
)

const DownloadIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
)

const InfoIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

interface PromptCardProps {
  prompt: any
  index: number
  onImageView: (prompt: any, index: number) => void
}

function PromptCard({ prompt, index, onImageView }: PromptCardProps) {
  const { showSuccess, showError } = useToast()

  const handleDownload = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (!prompt.image_url) return
    const link = document.createElement('a')
    link.href = prompt.image_url
    link.download = `${prompt.title || 'prompt'}.png`
    link.click()
    showSuccess('Download started!')
  }

  const handleImageClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onImageView(prompt, index)
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    })
  }

  return (
    <Card className="bg-[#161B22] border-[#30363D] overflow-hidden hover:border-[#00D4AA]/50 transition-all duration-200 group">
      {/* Image Section */}
      <div className="relative aspect-square overflow-hidden">
        <CachedImage
          src={prompt.thumbnail_url || prompt.image_url}
          alt={prompt.title}
          className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105 cursor-pointer"
          onClick={handleImageClick}
          loadingComponent={
            <div className="flex items-center justify-center w-full h-full">
              <div className="animate-pulse bg-gray-300 dark:bg-gray-600 w-8 h-8 rounded"></div>
            </div>
          }
        />
        
        {/* Hover Overlay */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all duration-200 flex items-center justify-center">
          <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-3">
            <Button
              size="lg"
              variant="secondary"
              className="bg-black/90 hover:bg-[#00D4AA] hover:text-black text-white border-0 px-4 py-2"
              onClick={handleImageClick}
            >
              <EyeIcon />
              <span className="ml-2 font-medium">Preview</span>
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="bg-black/60 hover:bg-black/80 text-white/70 hover:text-white border-0 p-2"
              onClick={handleDownload}
            >
              <DownloadIcon />
            </Button>
          </div>
        </div>

        {/* Favorite Badge */}
        <div className="absolute top-2 right-2">
          <div className="bg-black/80 rounded-full p-1.5">
            <HeartIcon filled={true} />
          </div>
        </div>
      </div>

      {/* Content Section */}
      <div className="p-4 space-y-3">
        {/* Title */}
        <h3 className="font-semibold text-white text-sm line-clamp-1">
          {prompt.title}
        </h3>

        {/* Prompt Preview */}
        <p className="text-gray-400 text-xs line-clamp-2 leading-relaxed">
          {prompt.prompt}
        </p>

        {/* Metadata */}
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>{formatDate(prompt.created_at)}</span>
          <Badge variant="secondary" className="bg-[#00D4AA]/20 text-[#00D4AA] text-xs">
            Saved
          </Badge>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between pt-2 border-t border-[#30363D]">
          <Link
            href={`/prompts/${prompt.id}`}
            className="text-xs text-[#00D4AA] hover:underline flex items-center gap-1"
          >
            <InfoIcon />
            View Details
          </Link>
          
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs text-[#00D4AA] hover:text-white hover:bg-[#00D4AA]/20"
              onClick={handleImageClick}
            >
              <EyeIcon />
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-2 text-xs text-gray-500 hover:text-gray-300"
              onClick={handleDownload}
            >
              <DownloadIcon />
            </Button>
          </div>
        </div>
      </div>
    </Card>
  )
}

function PromptsPageClient() {
  const [prompts, setPrompts] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<any | null>(null)
  const [showImageViewer, setShowImageViewer] = useState(false)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)

  // Image preloading is handled by SmartImage component automatically

  useEffect(() => {
    async function fetchPromptsData() {
      try {
        setLoading(true)
        const { data, error } = await getPrompts(100, 0)
        if (error) {
          setError(error)
          return
        }
        setPrompts(data?.prompts || [])
      } catch (err: any) {
        setError(err.message || 'Failed to load prompts')
      } finally {
        setLoading(false)
      }
    }

    fetchPromptsData()
  }, [])

  const handleImageView = (prompt: any, index?: number) => {
    // Find the index if not provided
    const promptIndex = index !== undefined ? index : prompts.findIndex(p => p.id === prompt.id)
    setCurrentImageIndex(promptIndex)
    
    // Debug logging
    console.log('ImageViewer URLs:', {
      title: prompt.title,
      thumbnail_url: prompt.thumbnail_url,
      image_url: prompt.image_url,
      image_r2_url: prompt.image_r2_url,
      using_url: prompt.image_url
    })
    
    // Convert prompt to ImageViewer format
    const imageViewerData = {
      id: prompt.id,
      url: prompt.image_url,
      fallbackUrl: prompt.image_r2_url, // Add fallback URL
      title: prompt.title,
      description: prompt.description || '',
      originalUrl: prompt.image_url,
      platform: 'foto-render',
      isPrivate: !prompt.is_public,
      isFavorite: true,
      userId: 'current-user',
      createdAt: prompt.created_at,
      updatedAt: prompt.created_at,
      tags: prompt.tags?.map((tag: string) => ({ name: tag, color: '#00D4AA' })) || [],
      parameters: {
        prompt: prompt.prompt,
        negative_prompt: prompt.negative_prompt || '',
        width: 1024,
        height: 1024,
        steps: 30,
        guidance_scale: 7.5,
        sampler: 'DPM++ 2M SDE Karras',
        model: 'Unknown',
        seed: -1,
        clip_skip: null,
      }
    }
    
    setSelectedImage(imageViewerData)
    setShowImageViewer(true)
  }

  const navigateImage = (direction: 'prev' | 'next') => {
    if (prompts.length === 0) return
    
    let newIndex = currentImageIndex
    if (direction === 'next') {
      newIndex = (currentImageIndex + 1) % prompts.length
    } else {
      newIndex = currentImageIndex === 0 ? prompts.length - 1 : currentImageIndex - 1
    }
    
    // Only update the index, don't change the selectedImage object
    // This prevents the ImageViewer from remounting
    setCurrentImageIndex(newIndex)
  }

  // Memoized image data for ImageViewer - only changes when URL actually changes
  const currentImageData = useMemo(() => {
    if (!selectedImage || !prompts[currentImageIndex]) return selectedImage
    
    const currentPrompt = prompts[currentImageIndex]
    const currentUrl = currentPrompt.image_url
    
    // Only create new object if URL is different
    if (selectedImage.url === currentUrl) {
      return selectedImage
    }
    
    return {
      ...selectedImage,
      url: currentUrl,
      title: currentPrompt.title,
      id: currentPrompt.id,
    }
  }, [selectedImage, prompts, currentImageIndex])

  // Keyboard navigation is now handled within the ImageViewer component

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="text-gray-400 mt-4">Loading your prompt library...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-500 text-lg font-semibold mb-2">Error Loading Prompts</div>
          <p className="text-gray-400">{error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-[#0D1117]">
      {/* Header */}
      <div className="bg-[#161B22] border-b border-[#30363D] px-4 sm:px-6 py-6">
        <div className="container mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                📚 Prompt Library
              </h1>
              <p className="text-gray-400 mt-1">
                {prompts.length} saved {prompts.length === 1 ? 'prompt' : 'prompts'}
              </p>
            </div>
            
            <div className="text-sm text-gray-400">
              Click images to preview • Use ← → arrows to navigate • View details for full info
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 sm:px-6 py-8">
        {prompts.length === 0 ? (
          <div className="text-center py-16">
            <div className="text-6xl mb-4">🎨</div>
            <h2 className="text-xl font-semibold text-white mb-2">No Prompts Saved Yet</h2>
            <p className="text-gray-400 mb-6">
              Start generating images and save your favorite prompts to build your library.
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 bg-[#00D4AA] text-black px-6 py-3 rounded-lg font-medium hover:bg-[#00D4AA]/90 transition-colors"
            >
              🎨 Start Generating
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-6">
            {prompts.map((prompt, index) => (
              <PromptCard
                key={prompt.id}
                prompt={prompt}
                index={index}
                onImageView={handleImageView}
              />
            ))}
          </div>
        )}
      </div>

      {/* Image Viewer Modal */}
      {showImageViewer && selectedImage && prompts.length > 0 && (
        <ImageViewer
          image={currentImageData}
          onClose={() => {
            setShowImageViewer(false)
            setSelectedImage(null)
          }}
          onToggleFavorite={() => {}} // No-op since already favorited
          onEdit={() => {}} // No-op for now
          onPrevious={prompts.length > 1 ? () => navigateImage('prev') : undefined}
          onNext={prompts.length > 1 ? () => navigateImage('next') : undefined}
          currentIndex={currentImageIndex}
          totalCount={prompts.length}
          hasPrevious={prompts.length > 1}
          hasNext={prompts.length > 1}
        />
      )}
    </div>
  )
}

export default function PromptsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-[#0D1117] flex items-center justify-center">
        <div className="text-center">
          <Spinner size="lg" />
          <p className="text-gray-400 mt-4">Loading prompt library...</p>
        </div>
      </div>
    }>
      <PromptsPageClient />
    </Suspense>
  )
} 
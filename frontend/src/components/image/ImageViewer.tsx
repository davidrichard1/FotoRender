'use client'

import { useState, useEffect, useRef } from 'react'
import NextImage from 'next/image'
import { useRouter } from 'next/navigation'
import { Image as ImageType } from '@/types/image'
import { SmartImage } from './SmartImage'

interface ImageViewerProps {
  image: ImageType
  onClose: () => void
  onToggleFavorite: (imageId: string, isFavorite: boolean) => void
  onEdit: (imageId: string) => void
  // Navigation props
  onPrevious?: () => void
  onNext?: () => void
  currentIndex?: number
  totalCount?: number
  hasPrevious?: boolean
  hasNext?: boolean
  // Details prop
  showDetailsDefault?: boolean
}

export function ImageViewer({
  image,
  onClose,
  onToggleFavorite,
  onEdit,
  onPrevious,
  onNext,
  currentIndex,
  totalCount,
  hasPrevious,
  hasNext,
  showDetailsDefault = false
}: ImageViewerProps) {
  const [showDetails, setShowDetails] = useState(showDetailsDefault)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageError, setImageError] = useState(false)
  const imageRef = useRef<HTMLImageElement>(null)
  const router = useRouter()

  const handleUsePrompt = () => {
    if (image.isPromptData && image.prompt) {
      // Store the prompt data in localStorage for the generate page to pick up
      const promptData = {
        positive_prompt: image.prompt,
        negative_prompt: image.negative_prompt || '',
        timestamp: Date.now()
      }
      localStorage.setItem('auto_fill_prompt', JSON.stringify(promptData))
      
      // Navigate to the generate page
      router.push('/')
      
      // Close the modal
      onClose()
    }
  }

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case 'i':
        case 'I':
          setShowDetails(!showDetails)
          break
        case 'f':
        case 'F':
          onToggleFavorite(image.id, !image.isFavorite)
          break
        case 'ArrowLeft':
          e.preventDefault()
          if (hasPrevious && onPrevious) {
            onPrevious()
          }
          break
        case 'ArrowRight':
          e.preventDefault()
          if (hasNext && onNext) {
            onNext()
          }
          break
        default:
          // No action for other keys
          break
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [image.id, image.isFavorite, showDetails, onClose, onToggleFavorite, onPrevious, onNext, hasPrevious, hasNext])

  // Prevent body scroll when modal is open
  useEffect(() => {
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [])

  const handleImageLoad = () => {
    console.log('ImageViewer: Image loaded successfully!', {
      imageId: image.id,
      title: image.title.substring(0, 30) + '...',
      url: image.url
    })
    setImageLoaded(true)
    setImageError(false)
  }

  const handleImageError = () => {
    console.error('ImageViewer: Image failed to load!', {
      imageId: image.id,
      title: image.title.substring(0, 30) + '...',
      url: image.url
    })
    setImageLoaded(false)
    setImageError(true)
  }

  const handleBackgroundClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  // Handle touch events for mobile swipe down to close
  const [touchStart, setTouchStart] = useState(0)

  const handleTouchStart = (e: React.TouchEvent) => {
    setTouchStart(e.touches[0].clientY)
  }

  const handleTouchEnd = (e: React.TouchEvent) => {
    const touchEnd = e.changedTouches[0].clientY
    const diff = touchStart - touchEnd

    // If swiped down more than 100px, close the modal
    if (diff < -100) {
      onClose()
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/95 backdrop-blur-sm"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* Enhanced Background overlay for easier closing */}
      <div
        className="absolute inset-0 cursor-pointer"
        onClick={handleBackgroundClick}
      />

      {/* Enhanced close button - elegant and prominent */}
      <button
        onClick={onClose}
        className="fixed top-4 right-4 z-[100] p-3 rounded-full bg-black/60 hover:bg-black/80 text-white transition-all duration-200 backdrop-blur-sm border border-white/20 shadow-xl group"
        title="Close (Escape)"
      >
        <svg
          className="w-6 h-6 transform group-hover:scale-110 transition-transform"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          strokeWidth={2.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>

      {/* Navigation Counter */}
      {typeof currentIndex === 'number' && typeof totalCount === 'number' && totalCount > 1 && (
        <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-[100] bg-black/80 backdrop-blur-sm text-white px-4 py-2 rounded-full text-sm border border-white/20">
          {currentIndex + 1} / {totalCount}
        </div>
      )}

      {/* Controls - simplified without delete button */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-sm border border-white/10"
          title="Toggle Details (I)"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </button>

        <button
          onClick={() => onToggleFavorite(image.id, !image.isFavorite)}
          className="p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-sm border border-white/10"
          title={`${image.isFavorite ? 'Remove from' : 'Add to'} Favorites (F)`}
        >
          <svg
            className="w-5 h-5"
            fill={image.isFavorite ? 'currentColor' : 'none'}
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
            />
          </svg>
        </button>

        <button
          onClick={() => onEdit(image.id)}
          className="p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-sm border border-white/10"
          title="Edit Image"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
            />
          </svg>
        </button>
      </div>

      {/* Navigation Arrows */}
      {hasPrevious && onPrevious && (
        <button
          onClick={onPrevious}
          className="fixed left-4 top-1/2 transform -translate-y-1/2 z-[90] p-4 rounded-full bg-black/60 hover:bg-black/80 text-white transition-all duration-200 backdrop-blur-sm border border-white/20 shadow-xl group"
          title="Previous Image (←)"
        >
          <svg
            className="w-8 h-8 transform group-hover:scale-110 transition-transform"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 19l-7-7 7-7"
            />
          </svg>
        </button>
      )}

      {hasNext && onNext && (
        <button
          onClick={onNext}
          className="fixed right-4 top-1/2 transform -translate-y-1/2 z-[90] p-4 rounded-full bg-black/60 hover:bg-black/80 text-white transition-all duration-200 backdrop-blur-sm border border-white/20 shadow-xl group"
          title="Next Image (→)"
        >
          <svg
            className="w-8 h-8 transform group-hover:scale-110 transition-transform"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 5l7 7-7 7"
            />
          </svg>
        </button>
      )}

      {/* Image container - truly maximized */}
      <div
        className="absolute inset-0 flex items-center justify-center p-2 md:p-4"
        style={{
          paddingTop: '60px',
          paddingBottom: showDetails ? '280px' : '60px'
        }}
      >
        <div className="relative w-full h-full">
          {!imageLoaded && !imageError && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white"></div>
            </div>
          )}

          {imageError && (
            <div className="absolute inset-0 flex items-center justify-center text-white">
              <div className="text-center">
                <svg
                  className="w-16 h-16 mx-auto mb-4 text-gray-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <p className="text-lg font-semibold">Failed to load image</p>
                <p className="text-sm text-gray-400 mt-1">
                  The image could not be displayed
                </p>
              </div>
            </div>
          )}

          {/* Use SmartImage for better CORS and caching handling */}
          <SmartImage
            ref={imageRef}
            src={image.url}
            fallbackSrc={image.fallbackUrl}
            alt={image.title}
            onLoad={handleImageLoad}
            onError={() => handleImageError()}
            className="absolute inset-0 w-full h-full object-contain"
            draggable={false}
            onClick={(e: React.MouseEvent) => e.stopPropagation()}
            forceNativeImg={true}
          />
        </div>
      </div>

      {/* Mobile swipe hint */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20 md:hidden">
        <div className="bg-black/50 backdrop-blur-sm rounded-full px-4 py-2 text-white text-sm border border-white/20">
          Swipe down to close
        </div>
      </div>

      {/* Details panel */}
      {showDetails && (
        <div className="absolute bottom-4 left-4 right-4 z-20">
          <div className="bg-black/80 backdrop-blur-sm rounded-xl p-6 text-white max-h-64 overflow-y-auto border border-white/20">
            {image.isPromptData ? (
              // Prompt-specific details
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-lg text-yellow-300">
                    {image.title}
                  </h3>
                  {/* Use Prompt Button */}
                  {image.prompt && (
                    <button
                      onClick={handleUsePrompt}
                      className="px-4 py-2 bg-[#00D4AA] text-black rounded-lg font-medium hover:bg-[#00D4AA]/90 transition-colors flex items-center gap-2 text-sm"
                      title="Use this prompt in the generator"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Use Prompt
                    </button>
                  )}
                </div>
                
                {/* Positive Prompt */}
                {image.prompt && (
                  <div>
                    <span className="text-gray-400 text-sm font-medium block mb-2">
                      Positive Prompt:
                    </span>
                    <p className="text-sm text-gray-200 bg-gray-900/50 rounded p-3 border border-gray-700">
                      {image.prompt}
                    </p>
                  </div>
                )}

                {/* Negative Prompt */}
                {image.negative_prompt && (
                  <div>
                    <span className="text-gray-400 text-sm font-medium block mb-2">
                      Negative Prompt:
                    </span>
                    <p className="text-sm text-gray-200 bg-gray-900/50 rounded p-3 border border-gray-700">
                      {image.negative_prompt}
                    </p>
                  </div>
                )}

                {/* Metadata */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-gray-700">
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Created:</span>
                      <span>{new Date(image.createdAt).toLocaleDateString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Platform:</span>
                      <span>{image.platform}</span>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-400 space-y-1">
                    <div className="font-medium text-gray-300">Controls:</div>
                    <div>• Desktop: Ctrl+scroll wheel to zoom</div>
                    <div>• Keyboard: I for details, Esc to close</div>
                    {(hasPrevious || hasNext) && (
                      <div>• Navigation: ← → arrow keys</div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              // Generic image details
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-lg mb-3 text-yellow-300">
                    {image.title}
                  </h3>
                  {image.description && (
                    <p className="text-sm text-gray-300 mb-4">
                      {image.description}
                    </p>
                  )}
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Created:</span>
                      <span>
                        {new Date(image.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Platform:</span>
                      <span>{image.platform}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Favorite:</span>
                      <span>{image.isFavorite ? '⭐ Yes' : 'No'}</span>
                    </div>
                  </div>
                </div>

                <div>
                  {image.tags.length > 0 && (
                    <div className="mb-4">
                      <span className="text-gray-400 text-sm mb-2 block">
                        Tags:
                      </span>
                      <div className="flex flex-wrap gap-2">
                        {image.tags.map((tag, index) => (
                          <span
                            key={index}
                            className="px-3 py-1 rounded-full text-xs font-medium border"
                            style={{
                              backgroundColor: `${tag.color}20`,
                              color: tag.color,
                              borderColor: `${tag.color}40`
                            }}
                          >
                            {tag.name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="text-xs text-gray-400 space-y-1">
                    <div className="font-medium text-gray-300">Controls:</div>
                    <div>• Desktop: Ctrl+scroll wheel to zoom</div>
                    <div>• Mobile: Pinch to zoom</div>
                    <div>• Keyboard: I for details, Esc to close</div>
                    {(hasPrevious || hasNext) && (
                      <div>• Navigation: ← → arrow keys</div>
                    )}
                    <div>• Mobile: Swipe down to close</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

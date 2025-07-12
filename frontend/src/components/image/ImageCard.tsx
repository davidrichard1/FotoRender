'use client'

import { useState, useRef, useEffect } from 'react'
import NextImage from 'next/image'
import { Image } from '@/types/image'

interface ImageCardProps {
  image: Image
  onView: (image: Image) => void
  onToggleFavorite: (imageId: string, isFavorite: boolean) => void
  onDelete: (imageId: string) => void
  onTagsUpdate: (
    imageId: string,
    tags: { name: string; color?: string }[]
  ) => void
  onEditImage: (image: Image) => void
  openDropdownId: string | null
  onDropdownToggle: (imageId: string | null) => void
  // Drag and drop props
  onDragStart?: (
    contentId: string,
    contentType: 'video' | 'image',
    currentFolderId?: string | null
  ) => void
  onDragEnd?: () => void
  isDragging?: boolean
  isMoving?: boolean
  // Selection props
  isSelected?: boolean
  onSelect?: (imageId: string, selected: boolean) => void
  showSelection?: boolean
}

export const ImageCard = ({
  image,
  onView,
  onToggleFavorite,
  onDelete,
  onEditImage,
  openDropdownId,
  onDropdownToggle,
  onDragStart,
  onDragEnd,
  isDragging = false,
  isMoving = false,
  isSelected = false,
  onSelect,
  showSelection = false
}: ImageCardProps) => {
  const [imageError, setImageError] = useState(false)
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0 })
  const dropdownRef = useRef<HTMLButtonElement>(null)

  const isDropdownOpen = openDropdownId === image.id

  // Drag and drop handlers
  const handleDragStart = (e: React.DragEvent) => {
    if (onDragStart) {
      onDragStart(image.id, 'image', image.folderId)
    }

    // Add global dragging cursor immediately
    document.body.style.cursor = 'grabbing'
    document.body.classList.add('dragging')

    // Set drag effect and visual feedback
    e.dataTransfer.effectAllowed = 'move'
    e.dataTransfer.dropEffect = 'move'
    e.dataTransfer.setData('text/plain', image.id)
    e.dataTransfer.setData('application/x-content-type', 'image')
    e.dataTransfer.setData('application/x-content-id', image.id)
    e.dataTransfer.setData('application/x-folder-id', image.folderId || '')

    // Create a simple drag image with better visibility
    const dragImg = document.createElement('div')
    dragImg.innerHTML = `
      <div style="
        background: #10B981; 
        color: white; 
        padding: 8px 12px; 
        border-radius: 8px; 
        font-size: 14px; 
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 2px solid #059669;
        display: flex;
        align-items: center;
        gap: 8px;
      ">
        📸 Moving "${(image.title || 'Untitled Image').substring(0, 30)}${
      (image.title || 'Untitled Image').length > 30 ? '...' : ''
    }"
      </div>
    `
    dragImg.style.position = 'absolute'
    dragImg.style.top = '-1000px'
    dragImg.style.left = '-1000px'
    dragImg.style.pointerEvents = 'none'
    document.body.appendChild(dragImg)

    e.dataTransfer.setDragImage(dragImg, 0, 0)

    // Clean up the drag image after a short delay
    setTimeout(() => {
      document.body.removeChild(dragImg)
    }, 0)
  }

  const handleDragEnd = () => {
    // Remove global dragging cursor and cleanup
    document.body.style.cursor = ''
    document.body.classList.remove('dragging')

    if (onDragEnd) {
      onDragEnd()
    }
  }

  // Get the image URL with fallback
  const getImageSrc = () => {
    if (imageError) return '/placeholder-image.png'

    // Use proxyUrl if available, otherwise fallback to url
    if (image.proxyUrl && image.proxyUrl.trim()) {
      return image.proxyUrl.trim()
    }

    return image.url.trim()
  }

  useEffect(() => {
    if (isDropdownOpen && dropdownRef.current) {
      const rect = dropdownRef.current.getBoundingClientRect()
      const viewportHeight = window.innerHeight
      const viewportWidth = window.innerWidth
      const dropdownHeight = 300 // Approximate dropdown height
      const dropdownWidth = 200 // Approximate dropdown width

      let top = rect.bottom + 8
      let { left } = rect

      // Adjust if dropdown would go off bottom of screen
      if (top + dropdownHeight > viewportHeight) {
        top = rect.top - dropdownHeight - 8
      }

      // Smart horizontal positioning - check if there's enough space on the right
      const spaceOnRight = viewportWidth - rect.right
      const spaceOnLeft = rect.left

      if (spaceOnRight < dropdownWidth && spaceOnLeft > dropdownWidth) {
        // Not enough space on right, but enough on left - align to right edge of button
        left = rect.right - dropdownWidth
      } else if (spaceOnRight < dropdownWidth && spaceOnLeft < dropdownWidth) {
        // Not enough space on either side - center on screen
        left = Math.max(8, (viewportWidth - dropdownWidth) / 2)
      }

      // Ensure dropdown doesn't go off left edge
      left = Math.max(8, left)

      // Ensure dropdown doesn't go off right edge
      left = Math.min(left, viewportWidth - dropdownWidth - 8)

      setDropdownPosition({ top, left })
    }
  }, [isDropdownOpen])

  // Click outside detection for dropdown
  useEffect(() => {
    if (!isDropdownOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        // Check if the click is outside the dropdown menu itself
        const dropdownMenu = document.querySelector(
          `[data-dropdown="${image.id}"]`
        )
        if (!dropdownMenu || !dropdownMenu.contains(event.target as Node)) {
          onDropdownToggle(null)
        }
      }
    }

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onDropdownToggle(null)
      }
    }

    // Use capture phase to ensure we catch the event before other handlers
    document.addEventListener('mousedown', handleClickOutside, true)
    document.addEventListener('keydown', handleEscape)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside, true)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [isDropdownOpen, image.id, onDropdownToggle])

  const handleImageError = () => {
    setImageError(true)
  }

  const handleDropdownToggle = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onDropdownToggle(isDropdownOpen ? null : image.id)
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const handleSelectionChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation()
    if (onSelect) {
      const { checked } = e.target
      onSelect(image.id, checked)
    }
  }

  // Ensure selection state is explicitly calculated
  const isCardSelected = showSelection && isSelected

  return (
    <div
      className={`group relative bg-white dark:bg-gray-800 rounded-lg shadow-sm hover:shadow-lg transition-all duration-200 border border-gray-200 dark:border-gray-700 ${
        isDragging ? 'opacity-50 scale-95 rotate-2 z-50' : ''
      } ${isMoving ? 'opacity-60 pointer-events-none' : ''} ${
        isCardSelected ? 'ring-2 ring-blue-500 border-blue-500' : ''
      }`}
    >
      {/* Moving Overlay */}
      {isMoving && (
        <div className="absolute inset-0 bg-blue-500/10 rounded-xl flex items-center justify-center z-40">
          <div className="bg-white/90 dark:bg-gray-800/90 px-4 py-2 rounded-lg shadow-lg border border-blue-200 dark:border-blue-700">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
                Moving...
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Selection Checkbox */}
      {showSelection && (
        <div className="absolute top-3 left-3 z-20 bg-white/90 dark:bg-gray-800/90 rounded-md p-1 backdrop-blur-sm border border-gray-200 dark:border-gray-600">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={handleSelectionChange}
            className="w-5 h-5 text-blue-600 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-500 rounded focus:ring-2 focus:ring-blue-500 cursor-pointer"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}

      {/* Drag Handle - Top right like video cards */}
      {!isDropdownOpen && (
        <div
          className="drag-handle absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300 z-30 p-1.5 bg-gray-900/80 rounded-md shadow-lg cursor-grab active:cursor-grabbing"
          draggable={true}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          title="Drag to move image"
        >
          <svg
            className="w-4 h-4 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 8h16M4 16h16"
            />
          </svg>
        </div>
      )}

      {/* Image Container */}
      <div
        className="relative aspect-square cursor-pointer overflow-hidden rounded-t-lg"
        onClick={() => onView(image)}
      >
        {!imageError ? (
          <NextImage
            src={getImageSrc()}
            alt={image.title || 'Image'}
            fill
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            className="object-cover transition-transform duration-300 group-hover:scale-[1.02]"
            onError={handleImageError}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-100 dark:bg-gray-700">
            <div className="text-center text-gray-500 dark:text-gray-400">
              <div className="text-4xl mb-2">🖼️</div>
              <div className="text-sm">Image unavailable</div>
            </div>
          </div>
        )}

        {/* Overlay with controls */}
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-300 flex flex-col justify-between p-3">
          {/* Top section - favorite button */}
          <div className="flex justify-start">
            <button
              onClick={(e) => {
                e.stopPropagation()
                onToggleFavorite(image.id, !image.isFavorite)
              }}
              className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 p-2 rounded-full bg-black bg-opacity-50 text-white hover:bg-opacity-70"
              aria-label={
                image.isFavorite ? 'Remove from favorites' : 'Add to favorites'
              }
            >
              <span className="text-lg">{image.isFavorite ? '❤️' : '🤍'}</span>
            </button>
          </div>

          {/* Bottom section - menu only */}
          <div className="flex justify-end items-end">
            {/* Menu button - bottom right */}
            <button
              ref={dropdownRef}
              onClick={handleDropdownToggle}
              className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 p-2 rounded-full bg-black bg-opacity-50 text-white hover:bg-opacity-70"
              aria-label="More options"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Status indicators */}
        <div className="absolute bottom-3 left-3 flex gap-2">
          {image.isPrivate && (
            <span className="px-2 py-1 text-xs bg-orange-500 text-white rounded-full shadow-sm">
              Private
            </span>
          )}
          {image.isFavorite && (
            <span className="px-2 py-1 text-xs bg-red-500 text-white rounded-full shadow-sm">
              ❤️
            </span>
          )}
        </div>
      </div>

      {/* Card content */}
      <div className="p-4">
        <h3 className="font-medium text-gray-900 dark:text-white truncate mb-2">
          {image.title || 'Untitled Image'}
        </h3>

        {image.description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-3">
            {image.description}
          </p>
        )}

        {/* Tags */}
        {image.tags && image.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {image.tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="inline-block px-2 py-1 text-xs rounded-full text-white"
                style={{
                  backgroundColor: tag.color || '#6366F1'
                }}
              >
                {tag.name}
              </span>
            ))}
            {image.tags.length > 3 && (
              <span className="inline-block px-2 py-1 text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-full">
                +{image.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* Metadata */}
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
          <span className="flex items-center gap-1">
            <span>📅</span>
            {new Date(image.createdAt).toLocaleDateString()}
          </span>
          {image.fileSize && (
            <span className="flex items-center gap-1">
              <span>📊</span>
              {formatFileSize(image.fileSize)}
            </span>
          )}
        </div>

        {image.width && image.height && (
          <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            {image.width} × {image.height}
          </div>
        )}
      </div>

      {/* Dropdown Menu Portal */}
      {isDropdownOpen && (
        <div
          className="fixed z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 py-2 min-w-[200px]"
          style={{
            top: dropdownPosition.top,
            left: dropdownPosition.left
          }}
          data-dropdown={image.id}
        >
          <button
            onClick={(e) => {
              e.stopPropagation()
              onView(image)
              onDropdownToggle(null)
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
          >
            <span>👁️</span>
            View Image
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation()
              onEditImage(image)
              onDropdownToggle(null)
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
          >
            <span>✏️</span>
            Edit Image
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation()
              onToggleFavorite(image.id, !image.isFavorite)
              onDropdownToggle(null)
            }}
            className="w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
          >
            <span>{image.isFavorite ? '💔' : '❤️'}</span>
            {image.isFavorite ? 'Remove from Favorites' : 'Add to Favorites'}
          </button>

          <div className="border-t border-gray-200 dark:border-gray-600 my-2" />

          <button
            onClick={(e) => {
              e.stopPropagation()
              if (
                confirm(
                  `Are you sure you want to delete "${
                    image.title || 'this image'
                  }"?`
                )
              ) {
                onDelete(image.id)
              }
              onDropdownToggle(null)
            }}
            className="w-full px-4 py-2 text-left text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-2"
          >
            <span>🗑️</span>
            Delete Image
          </button>
        </div>
      )}
    </div>
  )
}

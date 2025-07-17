import { useState, useEffect, forwardRef } from 'react'
import NextImage from 'next/image'
import { useCachedImage } from '@/lib/imageCache'

interface SmartImageProps extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, 'src'> {
  src: string
  fallbackSrc?: string  // Add fallback source for CORS failures
  alt: string
  width?: number
  height?: number
  fill?: boolean
  sizes?: string
  priority?: boolean
  fallback?: React.ReactNode
  loadingComponent?: React.ReactNode
  onCacheHit?: () => void
  onCacheMiss?: () => void
  // Force using regular img tag even for local images
  forceNativeImg?: boolean
}

export const SmartImage = forwardRef<HTMLImageElement, SmartImageProps>(
  ({ 
    src, 
    fallbackSrc,
    alt, 
    width,
    height,
    fill,
    sizes,
    priority,
    fallback, 
    loadingComponent, 
    onCacheHit, 
    onCacheMiss, 
    onLoad, 
    onError,
    className = '',
    forceNativeImg = false,
    ...props 
  }, ref) => {
      const { cachedUrl, isLoading } = useCachedImage(src)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageError, setImageError] = useState(false)
  const [isCacheHit, setIsCacheHit] = useState(false)
  const [currentSrc, setCurrentSrc] = useState<string>('')
  const [hasFallback, setHasFallback] = useState(false)
  const [hasTriedR2, setHasTriedR2] = useState(false)

  // Determine if this is an external image
  const isExternalImage = src.startsWith('http://') || src.startsWith('https://')
  const shouldUseNativeImg = forceNativeImg || isExternalImage || cachedUrl?.startsWith('blob:')

  // Set current source when cachedUrl changes
  useEffect(() => {
    if (cachedUrl && !hasFallback) {
      setCurrentSrc(cachedUrl)
      setHasTriedR2(false) // Reset when source changes
    }
  }, [cachedUrl, hasFallback])

  // Debug logging in development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log('SmartImage Debug:', {
        originalSrc: src,
        fallbackSrc,
        cachedUrl,
        currentSrc,
        isLoading,
        imageLoaded,
        imageError,
        isExternalImage,
        shouldUseNativeImg,
        hasFallback,
        hasTriedR2,
        alt: alt.substring(0, 30) + '...'
      })
    }
  }, [src, fallbackSrc, cachedUrl, currentSrc, isLoading, imageLoaded, imageError, isExternalImage, shouldUseNativeImg, hasFallback, hasTriedR2, alt])

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.log('SmartImage: Image loaded successfully!', {
      src: e.currentTarget.src,
      naturalWidth: e.currentTarget.naturalWidth,
      naturalHeight: e.currentTarget.naturalHeight,
      alt: alt.substring(0, 30) + '...',
      wasFallback: hasFallback
    })
    setImageLoaded(true)
    setImageError(false)
    
    // Determine if this was a cache hit
    const wasCacheHit = cachedUrl !== src && (cachedUrl?.startsWith('blob:') ?? false)
    setIsCacheHit(wasCacheHit)
    
    if (wasCacheHit && onCacheHit) {
      onCacheHit()
    } else if (!wasCacheHit && onCacheMiss) {
      onCacheMiss()
    }
    
    if (onLoad) {
      onLoad(e)
    }
  }

  const handleError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.error('SmartImage: Image failed to load!', {
      src: e.currentTarget.src,
      alt: alt.substring(0, 30) + '...',
      hasFallback,
      hasTriedR2,
      originalSrc: src,
      fallbackSrc
    })

    // If we have a fallback URL and haven't tried it yet, use it
    if (!hasFallback && fallbackSrc && fallbackSrc !== currentSrc) {
      console.log('SmartImage: Trying fallback URL:', fallbackSrc)
      setHasFallback(true)
      setImageError(false)
      setImageLoaded(false)
      setCurrentSrc(fallbackSrc)
      return
    }

    // If fallback failed and this is a CDN URL, try converting to R2 domain
    if (!hasTriedR2 && currentSrc.includes('foto.mylinkbuddy.com')) {
      console.log('SmartImage: Trying direct R2 domain fallback...')
      const r2Url = currentSrc.replace('https://foto.mylinkbuddy.com/', 'https://pub-f62e7fde038b425bb95a1e0a48bd3e17.r2.dev/')
      console.log('SmartImage: Converted to R2 URL:', r2Url)
      setHasTriedR2(true)
      setImageError(false)
      setImageLoaded(false)
      setCurrentSrc(r2Url)
      return
    }
    
    setImageError(true)
    setImageLoaded(false)
    
    if (onError) {
      onError(e)
    }
  }

  // Still loading from cache
  if (isLoading || !cachedUrl || !currentSrc) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 dark:bg-gray-800 ${className}`}>
        {loadingComponent || (
          <div className="animate-pulse flex flex-col items-center">
            <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded mb-2"></div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Loading...</div>
          </div>
        )}
      </div>
    )
  }

  // Error state
  if (imageError) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 dark:bg-gray-800 ${className}`}>
        {fallback || (
          <div className="text-center text-gray-500 dark:text-gray-400">
            <div className="text-2xl mb-1">🖼️</div>
            <div className="text-xs">Failed to load</div>
            <div className="text-xs text-red-500 mt-1">URL: {currentSrc?.substring(0, 50)}...</div>
            {hasFallback && (
              <div className="text-xs text-yellow-500 mt-1">Provider fallback failed</div>
            )}
            {hasTriedR2 && (
              <div className="text-xs text-orange-500 mt-1">R2 direct access failed</div>
            )}
          </div>
        )}
      </div>
    )
  }

  // Use native img tag for external images or blob URLs to avoid CORS issues
  if (shouldUseNativeImg) {
    return (
      <div className="relative">
        <img
          ref={ref}
          src={currentSrc}
          alt={alt}
          onLoad={handleLoad}
          onError={handleError}
          className={`transition-opacity duration-300 ${className}`}
          style={{ 
            opacity: imageLoaded ? 1 : 0,
            width: '100%',
            height: 'auto',
            maxHeight: '95vh',
            objectFit: 'contain'
          }}
          {...props}
        />
        
        {/* Cache hit indicator (for debugging) */}
        {process.env.NODE_ENV === 'development' && isCacheHit && (
          <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-1 rounded opacity-75">
            Cached
          </div>
        )}
        
        {/* Debug info in development */}
        {process.env.NODE_ENV === 'development' && (
          <div className="absolute bottom-1 left-1 bg-black/80 text-white text-xs px-1 rounded opacity-75 max-w-[200px] truncate">
            {hasTriedR2 ? 'R2 Direct' : hasFallback ? 'Fallback URL' : (currentSrc?.startsWith('blob:') ? 'Blob URL' : 'Direct URL')}
          </div>
        )}
        
        {/* Loading overlay */}
        {!imageLoaded && !imageError && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800">
            {loadingComponent || (
              <div className="animate-pulse">
                <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

    // Use Next.js Image component for local images
    return (
      <div className="relative">
        <NextImage
          ref={ref as any}
          src={cachedUrl}
          alt={alt}
          width={width}
          height={height}
          fill={fill}
          sizes={sizes}
          priority={priority}
          onLoad={handleLoad}
          onError={handleError}
          className={`transition-opacity duration-300 ${
            imageLoaded ? 'opacity-100' : 'opacity-0'
          } ${className}`}
          {...(props as any)}
        />
        
        {/* Cache hit indicator (for debugging) */}
        {process.env.NODE_ENV === 'development' && isCacheHit && (
          <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-1 rounded opacity-75 z-10">
            Cached
          </div>
        )}
        
        {/* Loading overlay */}
        {!imageLoaded && !imageError && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-800 z-10">
            {loadingComponent || (
              <div className="animate-pulse">
                <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }
)

SmartImage.displayName = 'SmartImage' 
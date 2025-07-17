import { useState, useEffect, forwardRef } from 'react'
import { useCachedImage } from '@/lib/imageCache'

interface CachedImageProps extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, 'src'> {
  src: string
  alt: string
  fallback?: React.ReactNode
  loadingComponent?: React.ReactNode
  onCacheHit?: () => void
  onCacheMiss?: () => void
}

export const CachedImage = forwardRef<HTMLImageElement, CachedImageProps>(
  ({ 
    src, 
    alt, 
    fallback, 
    loadingComponent, 
    onCacheHit, 
    onCacheMiss, 
    onLoad, 
    onError,
    className = '',
    ...props 
  }, ref) => {
    const { cachedUrl, isLoading } = useCachedImage(src)
    const [imageLoaded, setImageLoaded] = useState(false)
    const [imageError, setImageError] = useState(false)
    const [isCacheHit, setIsCacheHit] = useState(false)

    useEffect(() => {
      if (cachedUrl && cachedUrl !== src) {
        setIsCacheHit(true)
        onCacheHit?.()
      } else if (cachedUrl === src) {
        setIsCacheHit(false)
        onCacheMiss?.()
      }
    }, [cachedUrl, src, onCacheHit, onCacheMiss])

    const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
      setImageLoaded(true)
      setImageError(false)
      onLoad?.(e)
    }

    const handleError = (e: React.SyntheticEvent<HTMLImageElement>) => {
      console.warn('CachedImage load error:', src, e)
      setImageError(true)
      setImageLoaded(false)
      onError?.(e)
    }

    // Loading state
    if (isLoading || !cachedUrl) {
      return (
        <div className={`flex items-center justify-center bg-gray-100 dark:bg-gray-800 ${className}`}>
          {loadingComponent || (
            <div className="animate-pulse">
              <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded"></div>
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
            </div>
          )}
        </div>
      )
    }

    return (
      <div className="relative">
        <img
          ref={ref}
          src={cachedUrl}
          alt={alt}
          onLoad={handleLoad}
          onError={handleError}
          className={`transition-opacity duration-300 ${
            imageLoaded ? 'opacity-100' : 'opacity-0'
          } ${className}`}
          {...props}
        />
        
        {/* Cache hit indicator (for debugging) */}
        {process.env.NODE_ENV === 'development' && isCacheHit && (
          <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-1 rounded opacity-75">
            Cached
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
)

CachedImage.displayName = 'CachedImage' 
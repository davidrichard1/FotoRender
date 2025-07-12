'use client'

import { Button } from '@/components/ui'

interface EmptyStateProps {
  hasTagFilter: boolean
  onClearFilter: () => void
}

export const EmptyState = ({
  hasTagFilter,
  onClearFilter,
}: EmptyStateProps) => {
  if (hasTagFilter) {
    return (
      <div className="text-center py-16 px-8">
        <div className="max-w-md mx-auto">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            No videos match your filter
          </h3>
          <p className="text-gray-500 dark:text-gray-400 mb-6">
            Try adjusting your tag selection or clear the filter to see all
            videos.
          </p>
          <Button onClick={onClearFilter} variant="outline">
            Clear Filter
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="text-center py-16 px-8">
      <div className="max-w-md mx-auto">
        <div className="text-6xl mb-4">🎥</div>
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          Your video library is empty
        </h3>
        <p className="text-gray-500 dark:text-gray-400 mb-6">
          Add your first video by pasting a URL above. We support YouTube,
          Vimeo, and many other platforms.
        </p>
        <div className="text-sm text-gray-400 dark:text-gray-500">
          Tip: Videos will automatically extract thumbnails and metadata!
        </div>
      </div>
    </div>
  )
}

export const EmptyTagState = () => (
  <div className="text-center py-12 px-6">
    <div className="max-w-sm mx-auto">
      {/* Animated icon */}
      <div className="relative mb-6">
        <div className="text-5xl mb-2 animate-bounce">🏷️</div>
        <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-r from-amber-400 to-yellow-500 rounded-full flex items-center justify-center text-white text-xs font-bold animate-pulse">
            +
        </div>
      </div>

      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
        <span className="bg-gradient-to-r from-amber-600 to-yellow-500 bg-clip-text text-transparent">
            Start organizing with tags!
        </span>
      </h3>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6 leading-relaxed">
          Tags help you organize and find your videos quickly. Add your first
          video above, then come back here to create tags like:
      </p>

      {/* Example tags */}
      <div className="flex flex-wrap gap-2 justify-center mb-6">
        {['Tutorial', 'Favorites', 'Work', 'Entertainment'].map(
          (tag, index) => (
            <span
              key={tag}
              className="px-3 py-1 bg-gradient-to-r from-blue-100 to-purple-100 dark:from-blue-900/40 dark:to-purple-900/40 text-blue-700 dark:text-blue-300 rounded-full text-xs font-medium animate-pulse"
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              {tag}
            </span>
          ),
        )}
      </div>

      <div className="text-xs text-gray-500 dark:text-gray-400 bg-amber-50/50 dark:bg-amber-900/20 rounded-lg p-3 border border-amber-200/30 dark:border-amber-700/30">
          💡 <strong>Pro tip:</strong> Tags are automatically created when you
          add them to videos
      </div>
    </div>
  </div>
)

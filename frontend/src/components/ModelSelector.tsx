'use client'

import { use, useState } from 'react'
import type { ModelsResponse } from '@/lib/data-promises'

interface ModelSelectorProps {
  modelsPromise: Promise<ModelsResponse>
  currentModel: string
  onModelChange: (modelName: string) => void
  onModelSwitch: (modelName: string) => Promise<void>
  className?: string
}

export function ModelSelector({
  modelsPromise,
  currentModel,
  onModelChange,
  onModelSwitch,
  className = ''
}: ModelSelectorProps) {
  const [showModelSelection, setShowModelSelection] = useState(false)
  const [isSwitching, setIsSwitching] = useState(false)

  // Use React 19's use() hook - suspends until promise resolves
  const modelsData = use(modelsPromise)

  // Handle errors from the promise
  if (modelsData.error) {
    throw new Error(modelsData.error)
  }

  const handleModelSwitch = async (modelName: string) => {
    if (modelName === currentModel || isSwitching) return

    setIsSwitching(true)
    try {
      await onModelSwitch(modelName)
      onModelChange(modelName)
    } catch (error) {
      console.error('Failed to switch model:', error)
      // Error will be caught by error boundary
      throw error
    } finally {
      setIsSwitching(false)
      setShowModelSelection(false)
    }
  }

  const currentModelData = modelsData.models.find(
    (m) => m.filename === currentModel
  )

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setShowModelSelection(!showModelSelection)}
        disabled={isSwitching}
        className="w-full sm:w-auto flex items-center gap-2 px-3 py-1.5 text-sm text-gray-300 hover:text-white bg-[#21262D] hover:bg-[#30363D] rounded-md border border-[#30363D] transition-colors disabled:opacity-50"
      >
        <span className="truncate max-w-[150px] sm:max-w-48">
          {currentModelData?.display_name || 'Select Model'}
        </span>
        {isSwitching ? (
          <div className="w-4 h-4 border-2 border-gray-600 border-t-blue-500 rounded-full animate-spin flex-shrink-0" />
        ) : (
          <svg
            className="w-4 h-4 text-gray-500 flex-shrink-0"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        )}
      </button>

      {/* Model Selection Dropdown */}
      {showModelSelection && !isSwitching && (
        <div className="absolute top-full left-0 right-0 sm:left-auto sm:right-0 sm:w-80 mt-1 bg-[#161B22] border border-[#30363D] rounded-lg shadow-xl z-[100] max-w-[320px] sm:max-w-none">
          <div className="p-2 border-b border-[#30363D]">
            <div className="text-xs font-medium text-gray-400 uppercase tracking-wide">
              Available Models ({modelsData.models.length})
            </div>
          </div>
          <div className="max-h-80 overflow-y-auto">
            {modelsData.models.map((model) => (
              <button
                key={model.filename}
                onClick={() => handleModelSwitch(model.filename)}
                className={`w-full text-left p-3 hover:bg-[#21262D] transition-colors border-b border-[#21262D] last:border-b-0 ${
                  currentModel === model.filename ? 'bg-[#21262D]' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-white truncate">
                      {model.display_name}
                    </div>
                    <div className="text-xs text-gray-500 truncate">
                      {model.description || model.capability}
                    </div>
                    <div className="flex gap-1 mt-1 flex-wrap">
                      <span className="text-xs bg-[#30363D] text-gray-300 px-1 py-0.5 rounded">
                        {model.base_model}
                      </span>
                      <span className="text-xs bg-[#30363D] text-gray-300 px-1 py-0.5 rounded">
                        {model.file_size_mb
                          ? `${model.file_size_mb.toFixed(0)}MB`
                          : '?MB'}
                      </span>
                      {model.is_nsfw && (
                        <span className="text-xs bg-red-500/20 text-red-400 px-1 py-0.5 rounded">
                          NSFW
                        </span>
                      )}
                    </div>
                  </div>
                  {currentModel === model.filename && (
                    <div className="w-2 h-2 bg-blue-500 rounded-full ml-2 flex-shrink-0"></div>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

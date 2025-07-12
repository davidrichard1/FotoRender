'use client'

import { use } from 'react'
import type {
  UpscalersResponse,
  EmbeddingsResponse,
  VaesResponse,
} from '@/lib/data-promises'

// Upscaler Selector Component
interface UpscalerSelectorProps {
  upscalersPromise: Promise<UpscalersResponse>
  selectedUpscaler: string
  customUpscaleScale: number | null
  onUpscalerChange: (filename: string) => void
  onCustomScaleChange: (scale: number | null) => void
}

export function UpscalerSelector({
  upscalersPromise,
  selectedUpscaler,
  customUpscaleScale,
  onUpscalerChange,
  onCustomScaleChange,
}: UpscalerSelectorProps) {
  const upscalersData = use(upscalersPromise)

  if (upscalersData.error) {
    throw new Error(upscalersData.error)
  }

  const selectedUpscalerData = upscalersData.upscalers.find(
    (u) => u.filename === selectedUpscaler,
  )

  return (
    <div className="bg-[#0D1117] rounded p-4 border border-[#30363D] mb-4">
      <h3 className="text-white font-medium mb-4">AI Upscaling</h3>

      {upscalersData.upscalers.length > 0 ? (
        <>
          {/* Upscaler Selection */}
          <div className="mb-4">
            <label className="text-white font-medium mb-2 block">
              Select Upscaler
            </label>
            <select
              value={selectedUpscaler}
              onChange={(e) => onUpscalerChange(e.target.value)}
              className="w-full px-3 py-2 bg-[#161B22] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
            >
              <option value="">Choose an upscaler...</option>
              {upscalersData.upscalers.map((upscaler) => (
                <option key={upscaler.filename} value={upscaler.filename}>
                  {upscaler.model_name} ({upscaler.scale_factor}x) -{' '}
                  {upscaler.type}
                </option>
              ))}
            </select>
          </div>

          {/* Custom Scale */}
          {selectedUpscaler && (
            <div className="mb-4">
              <label className="text-white font-medium mb-2 block">
                Scale Factor (optional)
              </label>
              <input
                type="number"
                min="1"
                max="8"
                step="1"
                value={customUpscaleScale || ''}
                onChange={(e) => onCustomScaleChange(
                  e.target.value ? parseInt(e.target.value, 10) : null,
                )
                }
                placeholder="Default"
                className="w-full px-3 py-2 bg-[#161B22] border border-[#30363D] rounded text-white placeholder-gray-500 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
              />
              <p className="text-xs text-gray-400 mt-1">
                Leave empty to use model default (
                {selectedUpscalerData?.scale_factor || 4}x)
              </p>
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-8">
          <p className="text-gray-400">No upscalers found</p>
          <p className="text-sm text-gray-500 mt-2">
            Place upscaler models in the upscalers/ directory
          </p>
        </div>
      )}
    </div>
  )
}

// Embedding Selector Component
interface EmbeddingSelectorProps {
  embeddingsPromise: Promise<EmbeddingsResponse>
  onAddToPrompt: (triggerWord: string, isNegative?: boolean) => void
}

export function EmbeddingSelector({
  embeddingsPromise,
  onAddToPrompt,
}: EmbeddingSelectorProps) {
  const embeddingsData = use(embeddingsPromise)

  if (embeddingsData.error) {
    throw new Error(embeddingsData.error)
  }

  return (
    <div className="bg-[#0D1117] rounded p-4 border border-[#30363D] mb-4">
      <h3 className="text-white font-medium mb-4">Embeddings</h3>

      {embeddingsData.embeddings.length > 0 ? (
        <div className="space-y-3">
          {embeddingsData.embeddings.map((embedding) => (
            <div
              key={`${embedding.name}-${embedding.trigger_word}`}
              className="bg-[#161B22] border border-[#30363D] rounded p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-white">
                  {embedding.name}
                </span>
                <div className="flex gap-1">
                  <button
                    onClick={() => embedding.trigger_word
                      && onAddToPrompt(embedding.trigger_word)
                    }
                    disabled={!embedding.trigger_word}
                    className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded hover:bg-emerald-500/30 text-xs disabled:opacity-50 disabled:cursor-not-allowed border border-emerald-500/30"
                  >
                    Add to Prompt
                  </button>
                  <button
                    onClick={() => embedding.trigger_word
                      && onAddToPrompt(embedding.trigger_word, true)
                    }
                    disabled={!embedding.trigger_word}
                    className="px-2 py-1 bg-red-500/20 text-red-400 rounded hover:bg-red-500/30 text-xs disabled:opacity-50 disabled:cursor-not-allowed border border-red-500/30"
                  >
                    Add to Negative
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-400 mb-1">
                {embedding.trigger_word}
              </p>
              <p className="text-xs text-gray-500">{embedding.description}</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-8">
          <p className="text-gray-400">No embeddings found</p>
          <p className="text-sm text-gray-500 mt-2">
            Place your embedding files in the embeddings folder
          </p>
        </div>
      )}
    </div>
  )
}

// VAE Selector Component
interface VaeSelectorProps {
  vaesPromise: Promise<VaesResponse>
  currentVae: string
  onVaeLoad: (vaeFilename: string) => Promise<void>
  onVaeReset: () => Promise<void>
}

export function VaeSelector({
  vaesPromise,
  currentVae,
  onVaeLoad,
  onVaeReset,
}: VaeSelectorProps) {
  const vaesData = use(vaesPromise)

  if (vaesData.error) {
    throw new Error(vaesData.error)
  }

  return (
    <div className="bg-[#0D1117] rounded p-4 border border-[#30363D] mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-medium">VAE Models</h3>
        <button
          onClick={onVaeReset}
          className="px-3 py-1 bg-slate-500/20 text-slate-400 rounded hover:bg-slate-500/30 text-sm border border-slate-500/30"
        >
          Reset
        </button>
      </div>

      <div className="space-y-4">
        {/* Current VAE */}
        <div className="bg-[#161B22] border border-[#30363D] rounded p-3">
          <h4 className="text-sm font-medium text-gray-400 mb-2">
            Current VAE
          </h4>
          <div className="text-base font-medium text-white">
            {currentVae === 'default' ? 'Default (Built-in)' : currentVae}
          </div>
        </div>

        {/* VAE Selection */}
        {vaesData.vaes.length > 0 ? (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-gray-400">
              Available VAEs
            </h4>
            <div className="flex flex-col gap-2">
              {vaesData.vaes.map((vae) => (
                <div
                  key={`vae-${vae.name}-${vae.filename}`}
                  className={`border rounded-lg p-3 cursor-pointer transition-all ${
                    currentVae === vae.filename
                      ? 'bg-emerald-500/20 border-emerald-500/50'
                      : 'bg-[#161B22] border-[#30363D] hover:bg-[#21262D]'
                  }`}
                  onClick={() => onVaeLoad(vae.filename)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-white">
                      {vae.name}
                    </span>
                    {currentVae === vae.filename && (
                      <span className="text-xs bg-emerald-500/30 text-emerald-400 px-2 py-1 rounded">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-400 mb-1">{vae.filename}</p>
                  <p className="text-xs text-gray-500">
                    {vae.description || 'No description available'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-400">No VAE models found</p>
            <p className="text-sm text-gray-500 mt-2">
              Place your VAE files in the vaes folder
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

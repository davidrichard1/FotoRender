'use client'

import { use } from 'react'
import type { LorasResponse } from '@/lib/data-promises'

interface SelectedLora {
  filename: string
  scale: number
}

interface LoraSelectorProps {
  lorasPromise: Promise<LorasResponse>
  selectedLoras: SelectedLora[]
  onLoraAdd: (filename: string, scale?: number) => void
  onLoraRemove: (filename: string) => void
  onLoraScaleUpdate: (filename: string, scale: number) => void
  onClearAll: () => void
}

export function LoraSelector({
  lorasPromise,
  selectedLoras,
  onLoraAdd,
  onLoraRemove,
  onLoraScaleUpdate,
  onClearAll,
}: LoraSelectorProps) {
  // Use React 19's use() hook - suspends until promise resolves
  const lorasData = use(lorasPromise)

  // Handle errors from the promise
  if (lorasData.error) {
    throw new Error(lorasData.error)
  }

  const availableLoras = lorasData.loras.filter(
    (lora) => !selectedLoras.find((selected) => selected.filename === lora.filename),
  )

  const getLoraConfig = (filename: string) => {
    const dbLora = lorasData.loras.find((lora) => lora.filename === filename)
    if (dbLora) {
      return {
        displayName: dbLora.display_name,
        minScale: dbLora.scale_config.min_scale,
        maxScale: dbLora.scale_config.max_scale,
        recommendedScale: dbLora.scale_config.recommended_scale,
        step: 0.01,
        description: dbLora.usage_tips.description,
        usageTips: dbLora.usage_tips.trigger_words || [],
      }
    }
    return null
  }

  return (
    <div className="bg-[#0D1117] rounded p-4 border border-[#30363D] mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-white font-medium flex items-center gap-2">
          LoRA Models ({selectedLoras.length} selected)
        </h3>
        {selectedLoras.length > 0 && (
          <button
            onClick={onClearAll}
            className="px-2 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-sm transition-colors border border-red-600/30"
          >
            Clear All
          </button>
        )}
      </div>

      {/* LoRA Selection Dropdown */}
      {availableLoras.length > 0 && (
        <div className="mb-4">
          <select
            onChange={(e) => {
              if (e.target.value) {
                onLoraAdd(e.target.value)
                e.target.value = ''
              }
            }}
            className="w-full px-3 py-2 bg-[#161B22] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
          >
            <option value="">+ Add LoRA...</option>
            {availableLoras.map((lora) => (
              <option key={lora.filename} value={lora.filename}>
                {lora.display_name} ({lora.size_mb}MB)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Selected LoRAs */}
      {selectedLoras.length > 0 ? (
        <div className="space-y-4">
          {selectedLoras.map((selectedLora) => {
            const loraInfo = lorasData.loras.find(
              (lora) => lora.filename === selectedLora.filename,
            )
            const loraConfig = getLoraConfig(selectedLora.filename)

            return (
              <div
                key={selectedLora.filename}
                className="p-3 bg-[#161B22] rounded border border-[#30363D]"
              >
                {/* Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-white truncate text-sm">
                      {loraConfig?.displayName
                        || loraInfo?.display_name
                        || selectedLora.filename}
                    </div>
                    <div className="text-xs text-gray-400">
                      {loraInfo?.size_mb}MB •{' '}
                      {loraInfo?.category?.replace('_', ' ') || 'general'}
                    </div>
                  </div>
                  <button
                    onClick={() => onLoraRemove(selectedLora.filename)}
                    className="px-2 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-xs transition-colors border border-red-600/30 ml-3"
                    title="Remove LoRA"
                  >
                    Remove
                  </button>
                </div>

                {/* Description */}
                {(loraConfig?.description
                  || loraInfo?.usage_tips?.description) && (
                  <div className="text-xs text-gray-400 mb-3 p-2 bg-[#0D1117] rounded border border-[#30363D]">
                    {loraConfig?.description
                      || loraInfo?.usage_tips.description}
                  </div>
                )}

                {/* Usage Tips */}
                {loraConfig?.usageTips && loraConfig.usageTips.length > 0 && (
                  <div className="mb-3 space-y-1">
                    {loraConfig.usageTips.map((tip: string, index: number) => (
                      <div
                        key={index}
                        className="text-xs text-amber-400 bg-amber-500/10 p-2 rounded border border-amber-500/20"
                      >
                        {tip}
                      </div>
                    ))}
                  </div>
                )}

                {/* Scale Controls */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-xs text-gray-400">
                    <span>
                      Scale:{' '}
                      {selectedLora.scale.toFixed(
                        loraConfig?.step === 0.01 ? 2 : 1,
                      )}
                    </span>
                    <span>
                      Range:{' '}
                      {loraConfig?.minScale
                        ?? loraInfo?.scale_config?.min_scale
                        ?? -5}{' '}
                      to{' '}
                      {loraConfig?.maxScale
                        ?? loraInfo?.scale_config?.max_scale
                        ?? 5}
                      {(loraConfig?.recommendedScale
                        ?? loraInfo?.scale_config?.recommended_scale) && (
                        <span className="text-blue-400 ml-2">
                          (Rec:{' '}
                          {loraConfig?.recommendedScale
                            ?? loraInfo?.scale_config.recommended_scale}
                          )
                        </span>
                      )}
                    </span>
                  </div>

                  {/* Slider */}
                  <div className="px-1">
                    <input
                      type="range"
                      min={
                        loraConfig?.minScale
                        ?? loraInfo?.scale_config?.min_scale
                        ?? -5
                      }
                      max={
                        loraConfig?.maxScale
                        ?? loraInfo?.scale_config?.max_scale
                        ?? 5
                      }
                      step={loraConfig?.step ?? 0.01}
                      value={selectedLora.scale}
                      onChange={(e) => onLoraScaleUpdate(
                        selectedLora.filename,
                        parseFloat(e.target.value),
                      )
                      }
                      className="w-full h-2 bg-[#30363D] rounded appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Control buttons and input */}
                  <div className="flex items-center gap-2">
                    {/* Quick preset buttons */}
                    <div className="flex gap-1">
                      {[
                        loraConfig?.minScale
                          ?? loraInfo?.scale_config?.min_scale
                          ?? -5,
                        ...(loraConfig ? [0.1] : [-1]),
                        0,
                        loraConfig?.recommendedScale
                          ?? loraInfo?.scale_config?.recommended_scale
                          ?? 1,
                        loraConfig?.maxScale
                          ?? loraInfo?.scale_config?.max_scale
                          ?? 5,
                      ]
                        .filter(
                          (value, index, arr) => arr.indexOf(value) === index,
                        )
                        .map((presetValue) => (
                          <button
                            key={presetValue}
                            onClick={() => onLoraScaleUpdate(
                              selectedLora.filename,
                              presetValue,
                            )
                            }
                            className={`px-2 py-1 text-xs rounded-lg border transition-colors ${
                              selectedLora.scale === presetValue
                                ? 'bg-blue-600 text-white border-blue-600'
                                : 'bg-[#21262D] text-gray-400 border-[#30363D] hover:bg-[#30363D] hover:text-gray-300'
                            }`}
                            title={(() => {
                              const recommendedScale = loraConfig?.recommendedScale
                                ?? loraInfo?.scale_config?.recommended_scale
                                ?? 1
                              return presetValue === recommendedScale
                                ? 'Recommended'
                                : `Set to ${presetValue}`
                            })()}
                          >
                            {(() => {
                              if (presetValue === 0) return '0'
                              if (presetValue > 0) return `+${presetValue}`
                              return presetValue
                            })()}
                          </button>
                        ))}
                    </div>

                    {/* Precise input */}
                    <input
                      type="number"
                      min={
                        loraConfig?.minScale
                        ?? loraInfo?.scale_config?.min_scale
                        ?? -5
                      }
                      max={
                        loraConfig?.maxScale
                        ?? loraInfo?.scale_config?.max_scale
                        ?? 5
                      }
                      step={loraConfig?.step ?? 0.01}
                      value={selectedLora.scale}
                      onChange={(e) => onLoraScaleUpdate(
                        selectedLora.filename,
                        parseFloat(e.target.value),
                      )
                      }
                      className="w-20 px-2 py-1 text-xs bg-[#0D1117] border border-[#30363D] rounded text-white focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-colors"
                    />

                    {/* Reset button */}
                    <button
                      onClick={() => onLoraScaleUpdate(
                        selectedLora.filename,
                        loraConfig?.recommendedScale
                            ?? loraInfo?.scale_config?.recommended_scale
                            ?? 1,
                      )
                      }
                      className="px-2 py-1 text-xs bg-[#21262D] hover:bg-[#30363D] text-gray-400 hover:text-gray-300 rounded border border-[#30363D] transition-colors"
                      title="Reset to recommended scale"
                      disabled={
                        selectedLora.scale
                        === (loraConfig?.recommendedScale
                          ?? loraInfo?.scale_config?.recommended_scale
                          ?? 1)
                      }
                    >
                      Reset
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-8">
          <p className="text-gray-400">No LoRAs selected</p>
          <p className="text-sm text-gray-500 mt-2">
            {availableLoras.length > 0
              ? 'Choose from the dropdown above to add LoRAs'
              : 'No LoRAs available for the current model'}
          </p>
        </div>
      )}

      {/* Quick Actions */}
      {availableLoras.length > 0 && (
        <div className="mt-4 pt-4 border-t border-[#30363D]">
          <div className="text-xs font-medium text-gray-300 mb-2">
            Quick Actions:
          </div>
          <div className="flex flex-wrap gap-2">
            {/* EasyNegative Quick Add */}
            {availableLoras.find((lora) => lora.filename.toLowerCase().includes('easynegative')) && (
              <button
                onClick={() => {
                  const easyNeg = availableLoras.find((lora) => lora.filename.toLowerCase().includes('easynegative'))
                  if (easyNeg) onLoraAdd(easyNeg.filename)
                }}
                className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded text-xs font-medium transition-colors border border-red-500/30"
              >
                Add EasyNegative
              </button>
            )}
          </div>
        </div>
      )}

      {/* LoRA Tips */}
      {selectedLoras.length === 0 && availableLoras.length > 0 && (
        <div className="text-xs text-gold-400 mt-3">
          💡 Tip: LoRAs support -5 to +5 range with 0.01 precision. Typical
          usage is 0.5-1.5, but detail-enhancing LoRAs often work best at
          0.1-0.2!
        </div>
      )}
    </div>
  )
}

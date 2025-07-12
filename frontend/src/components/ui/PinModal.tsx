import React, { useState, useRef, useEffect } from 'react'

interface PinModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
  mode: 'setup' | 'verify' | 'change'
  title?: string
}

export const PinModal: React.FC<PinModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
  mode,
  title,
}) => {
  const [pin, setPin] = useState('')
  const [currentPin, setCurrentPin] = useState('')
  const [confirmPin, setConfirmPin] = useState('')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [step, setStep] = useState<'current' | 'new' | 'confirm'>('current')

  const inputRefs = useRef<(HTMLInputElement | null)[]>([])

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setPin('')
      setCurrentPin('')
      setConfirmPin('')
      setError('')
      setStep(
        (() => {
          if (mode === 'setup') return 'new'
          if (mode === 'change') return 'current'
          return 'current'
        })(),
      )
    }
  }, [isOpen, mode])

  // Focus first input when modal opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        inputRefs.current[0]?.focus()
      }, 100)
    }
  }, [isOpen, step])

  const handleDigitInput = (
    index: number,
    value: string,
    currentValue: string,
  ) => {
    if (value.length > 1) return // Prevent pasting multiple digits

    if (!/^\d?$/.test(value)) return // Only allow digits

    const newPin = currentValue.split('')
    newPin[index] = value
    const updatedPin = newPin.join('').slice(0, 4)
    updateCurrentPin(updatedPin)
    setError('')

    // Auto-focus next input
    if (value && index < 3) {
      inputRefs.current[index + 1]?.focus()
    }
  }

  const handleKeyDown = (
    index: number,
    e: React.KeyboardEvent,
    currentValue: string,
  ) => {
    if (e.key === 'Backspace' && !currentValue[index] && index > 0) {
      inputRefs.current[index - 1]?.focus()
    }

    if (e.key === 'ArrowLeft' && index > 0) {
      inputRefs.current[index - 1]?.focus()
    }

    if (e.key === 'ArrowRight' && index < 3) {
      inputRefs.current[index + 1]?.focus()
    }

    if (e.key === 'Enter' && currentValue.length === 4) {
      handleSubmit()
    }
  }

  const getCurrentPin = () => {
    switch (step) {
    case 'current':
      return mode === 'verify' ? pin : currentPin
    case 'new':
      return pin
    case 'confirm':
      return confirmPin
    default:
      return pin
    }
  }

  const updateCurrentPin = (value: string) => {
    switch (step) {
    case 'current':
      if (mode === 'verify') setPin(value)
      else setCurrentPin(value)
      break
    case 'new':
      setPin(value)
      break
    case 'confirm':
      setConfirmPin(value)
      break
    default:
      // No action for unknown steps
      break
    }
  }

  const handleSubmit = async () => {
    const currentPinValue = getCurrentPin()

    if (currentPinValue.length !== 4) {
      setError('Please enter a 4-digit PIN')
      return
    }

    setIsLoading(true)
    setError('')

    try {
      if (mode === 'verify') {
        // Verify existing PIN
        const response = await fetch('/api/user/pin', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pin: currentPinValue }),
        })

        const result = await response.json()

        if (!response.ok) {
          setError(result.error || 'Invalid PIN')
          return
        }

        if (result.valid) {
          onSuccess()
        } else {
          setError('Invalid PIN')
        }
      } else if (mode === 'setup') {
        if (step === 'new') {
          setStep('confirm')
        } else if (step === 'confirm') {
          if (pin !== confirmPin) {
            setError('PINs do not match')
            return
          }

          // Set new PIN
          const response = await fetch('/api/user/pin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pin: confirmPin }),
          })

          const result = await response.json()

          if (!response.ok) {
            setError(result.error || 'Failed to set PIN')
            return
          }

          onSuccess()
        }
      } else if (mode === 'change') {
        if (step === 'current') {
          // Verify current PIN first
          const response = await fetch('/api/user/pin', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pin: currentPin }),
          })

          const result = await response.json()

          if (!response.ok || !result.valid) {
            setError('Current PIN is incorrect')
            return
          }

          setStep('new')
        } else if (step === 'new') {
          setStep('confirm')
        } else if (step === 'confirm') {
          if (pin !== confirmPin) {
            setError('PINs do not match')
            return
          }

          // Update PIN
          const response = await fetch('/api/user/pin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pin: confirmPin, currentPin }),
          })

          const result = await response.json()

          if (!response.ok) {
            setError(result.error || 'Failed to update PIN')
            return
          }

          onSuccess()
        }
      }
    } catch (err) {
      console.error('PIN operation error:', err)
      setError('Something went wrong. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const getTitle = () => {
    if (title) return title

    if (mode === 'setup') {
      return step === 'confirm' ? 'Confirm Your PIN' : 'Set Up Your PIN'
    }
    if (mode === 'change') {
      if (step === 'current') return 'Enter Current PIN'
      if (step === 'new') return 'Enter New PIN'
      return 'Confirm New PIN'
    }
    return 'Enter Your PIN'
  }

  const getDescription = () => {
    if (mode === 'setup') {
      return step === 'confirm'
        ? 'Please confirm your 4-digit PIN'
        : 'Create a 4-digit PIN to secure your private folders'
    }
    if (mode === 'change') {
      if (step === 'current') return 'Enter your current PIN to continue'
      if (step === 'new') return 'Enter your new 4-digit PIN'
      return 'Confirm your new PIN'
    }
    return 'Enter your PIN to access private content'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-md w-full p-6">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-blue-600 dark:text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 15v2m0 0v2m0-2h2m-2 0h-2m9-6a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            {getTitle()}
          </h2>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            {getDescription()}
          </p>
        </div>

        {/* PIN Input */}
        <div className="flex justify-center gap-3 mb-6">
          {[0, 1, 2, 3].map((index) => (
            <input
              key={`${step}-${index}`}
              ref={(el) => {
                inputRefs.current[index] = el
              }}
              type="password"
              inputMode="numeric"
              maxLength={1}
              value={getCurrentPin()[index] || ''}
              onChange={(e) => handleDigitInput(index, e.target.value, getCurrentPin())
              }
              onKeyDown={(e) => handleKeyDown(index, e, getCurrentPin())}
              className="w-12 h-12 text-center text-lg font-semibold border-2 border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-colors"
              autoComplete="off"
            />
          ))}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-red-600 dark:text-red-400 text-sm text-center">
              {error}
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={isLoading || getCurrentPin().length !== 4}
            className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center"
          >
            {(() => {
              if (isLoading) {
                return (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                )
              }
              if (step === 'current' && mode !== 'verify') {
                return 'Continue'
              }
              if (step === 'new' && mode !== 'verify') {
                return 'Continue'
              }
              return 'Confirm'
            })()}
          </button>
        </div>
      </div>
    </div>
  )
}

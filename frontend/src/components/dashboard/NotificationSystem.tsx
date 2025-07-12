'use client'

import { useEffect, useState } from 'react'
import { NotificationState } from '@/types/image'

interface NotificationSystemProps {
  notification: NotificationState
  onDismiss: () => void
}

export const NotificationSystem = ({
  notification,
  onDismiss,
}: NotificationSystemProps) => {
  const [isClosing, setIsClosing] = useState(false)

  // Helper functions to avoid nested ternary expressions
  const getNotificationBgClass = (type: string) => {
    switch (type) {
    case 'success':
      return 'bg-green-50 dark:bg-green-900 border-green-200 dark:border-green-700 text-green-800 dark:text-green-200'
    case 'error':
      return 'bg-red-50 dark:bg-red-900 border-red-200 dark:border-red-700 text-red-800 dark:text-red-200'
    default:
      return 'bg-blue-50 dark:bg-blue-900 border-blue-200 dark:border-blue-700 text-blue-800 dark:text-blue-200'
    }
  }

  const getIconBgClass = (type: string) => {
    switch (type) {
    case 'success':
      return 'bg-green-500'
    case 'error':
      return 'bg-red-500'
    default:
      return 'bg-blue-500'
    }
  }

  const getButtonTextClass = (type: string) => {
    switch (type) {
    case 'success':
      return 'text-green-600 dark:text-green-400'
    case 'error':
      return 'text-red-600 dark:text-red-400'
    default:
      return 'text-blue-600 dark:text-blue-400'
    }
  }

  useEffect(() => {
    if (notification.isVisible) {
      // Reset closing state for new notifications
      setIsClosing(false)

      // Simple auto-dismiss timer
      const dismissTime = notification.type === 'success' ? 3000 : 5000 // 3s for success, 5s for errors
      const fadeOutTime = 300 // 300ms fade out

      const timer = setTimeout(() => {
        // Start fade out animation
        setIsClosing(true)

        // Actually dismiss after fade completes
        setTimeout(() => {
          onDismiss()
        }, fadeOutTime)
      }, dismissTime - fadeOutTime)

      return () => clearTimeout(timer)
    }
    setIsClosing(false)
  }, [notification.id, notification.isVisible, notification.type, onDismiss])

  // Handle manual dismiss with fade
  const handleManualDismiss = () => {
    setIsClosing(true)
    setTimeout(() => {
      onDismiss()
    }, 300)
  }

  if (!notification.isVisible) return null

  const isSuccess = notification.type === 'success'
  const isError = notification.type === 'error'

  return (
    <div className="fixed top-4 right-4 z-50 max-w-sm">
      <div
        className={`
          px-6 py-4 rounded-lg shadow-lg border transition-all duration-300 ease-in-out transform
          ${getNotificationBgClass(notification.type)}
          ${
    isClosing
      ? 'opacity-0 scale-95 translate-x-2'
      : 'opacity-100 scale-100 translate-x-0 animate-in slide-in-from-right-full'
    }
        `}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div
              className={`w-5 h-5 rounded-full mr-3 flex items-center justify-center ${getIconBgClass(
                notification.type,
              )}`}
            >
              {isSuccess && (
                <svg
                  className="w-3 h-3 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
              {isError && (
                <svg
                  className="w-3 h-3 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </div>
            <p className="font-medium">{notification.message}</p>
          </div>
          <button
            onClick={handleManualDismiss}
            className={`ml-4 text-lg font-bold hover:opacity-70 transition-opacity ${getButtonTextClass(
              notification.type,
            )}`}
          >
            ×
          </button>
        </div>
      </div>
    </div>
  )
}

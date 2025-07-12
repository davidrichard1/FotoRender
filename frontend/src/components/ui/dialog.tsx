import React, { useEffect } from 'react'

export interface DialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  children: React.ReactNode
}

export interface DialogContentProps {
  children: React.ReactNode
  className?: string
}

export interface DialogHeaderProps {
  children: React.ReactNode
}

export interface DialogTitleProps {
  children: React.ReactNode
  className?: string
}

export interface DialogDescriptionProps {
  children: React.ReactNode
}

export interface DialogFooterProps {
  children: React.ReactNode
}

export const Dialog: React.FC<DialogProps> = ({
  open,
  onOpenChange,
  children,
}) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onOpenChange(false)
      }
    }

    if (open) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [open, onOpenChange])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50"
        onClick={() => onOpenChange(false)}
      />
      {/* Dialog */}
      <div className="relative z-50">{children}</div>
    </div>
  )
}

export const DialogContent: React.FC<DialogContentProps> = ({
  children,
  className = '',
}) => (
  <div
    className={`bg-white dark:bg-gray-800 rounded-lg shadow-xl border max-w-lg w-full mx-4 ${className}`}
  >
    {children}
  </div>
)

export const DialogHeader: React.FC<DialogHeaderProps> = ({ children }) => <div className="p-6 pb-2">{children}</div>

export const DialogTitle: React.FC<DialogTitleProps> = ({
  children,
  className = '',
}) => (
  <h2
    className={`text-lg font-semibold text-gray-900 dark:text-white ${className}`}
  >
    {children}
  </h2>
)

export const DialogDescription: React.FC<DialogDescriptionProps> = ({
  children,
}) => (
  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{children}</p>
)

export const DialogFooter: React.FC<DialogFooterProps> = ({ children }) => <div className="p-6 pt-2 flex justify-end gap-2">{children}</div>

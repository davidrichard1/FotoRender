import React from 'react'
import { cn } from '@/lib/utils'

// Extended types for our component
type ExtendedButtonVariant =
  | 'primary'
  | 'secondary'
  | 'outline'
  | 'ghost'
  | 'destructive'
type ExtendedButtonSize = 'sm' | 'md' | 'lg' | 'icon'

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ExtendedButtonVariant
  size?: ExtendedButtonSize
  loading?: boolean
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      children,
      disabled,
      className,
      ...props
    },
    ref,
  ) => {
    const getVariantClasses = (v: ExtendedButtonVariant) => {
      const variants = {
        primary:
          'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 dark:bg-blue-600 dark:hover:bg-blue-700',
        secondary:
          'bg-gray-200 text-gray-900 hover:bg-gray-300 active:bg-gray-400 dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600',
        outline:
          'border border-gray-300 bg-transparent text-gray-900 hover:bg-gray-50 active:bg-gray-100 dark:border-gray-600 dark:text-gray-100 dark:hover:bg-gray-800',
        ghost:
          'bg-transparent text-gray-900 hover:bg-gray-100 active:bg-gray-200 dark:text-gray-100 dark:hover:bg-gray-800',
        destructive:
          'bg-red-600 text-white hover:bg-red-700 active:bg-red-800 dark:bg-red-600 dark:hover:bg-red-700',
      }
      return variants[v]
    }

    const getSizeClasses = (s: ExtendedButtonSize) => {
      const sizes = {
        sm: 'px-3 py-1.5 text-sm h-8',
        md: 'px-4 py-2 text-base h-10',
        lg: 'px-6 py-2.5 text-lg h-12',
        icon: 'w-10 h-10 p-0',
      }
      return sizes[s]
    }

    const baseClasses = 'inline-flex items-center justify-center rounded-md font-medium cursor-pointer transition-all duration-200 select-none outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed'

    const buttonClasses = cn(
      baseClasses,
      getVariantClasses(variant),
      getSizeClasses(size),
      className,
    )

    return (
      <button
        className={buttonClasses}
        ref={ref}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg
            className="animate-spin w-4 h-4 mr-2"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </button>
    )
  },
)

Button.displayName = 'Button'

export { Button }

import React from 'react'
import { cn } from '@/lib/utils'

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'> {
  variant?: 'default' | 'error'
  inputSize?: 'sm' | 'md' | 'lg'
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      variant = 'default', inputSize = 'md', className, type, ...props
    },
    ref,
  ) => {
    const baseClasses = 'w-full border rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder:text-gray-600 dark:placeholder:text-gray-400 transition-all duration-200 outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed font-medium'

    const variantClasses = {
      default:
        'border-gray-300 dark:border-gray-600 focus:border-amber-500 focus:ring-amber-200 dark:focus:ring-amber-800 hover:border-amber-300 dark:hover:border-amber-600 focus:shadow-lg focus:shadow-amber-500/20',
      error:
        'border-red-500 focus:border-red-500 focus:ring-red-200 dark:focus:ring-red-800',
    }

    const sizeClasses = {
      sm: 'px-3 py-2 text-sm',
      md: 'px-4 py-3 text-base',
      lg: 'px-4 py-3 text-lg',
    }

    // Enhanced styling for password inputs to improve asterisk visibility
    const passwordClasses = type === 'password'
      ? 'font-mono tracking-wide text-gray-900 dark:text-gray-100 font-bold'
      : ''

    return (
      <input
        type={type}
        className={cn(
          baseClasses,
          variantClasses[variant],
          sizeClasses[inputSize],
          passwordClasses,
          className,
        )}
        ref={ref}
        {...props}
      />
    )
  },
)

Input.displayName = 'Input'

export { Input }

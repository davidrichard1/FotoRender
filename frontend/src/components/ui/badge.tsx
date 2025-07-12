import React from 'react'
import { cn } from '@/lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline'
}

const Badge = React.forwardRef<HTMLDivElement, BadgeProps>(
  ({ variant = 'default', className, ...props }, ref) => {
    const variants = {
      default: 'bg-blue-500 text-white dark:bg-blue-600',
      secondary:
        'bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-100',
      destructive: 'bg-red-500 text-white dark:bg-red-600',
      outline:
        'border border-gray-300 bg-transparent text-gray-900 dark:border-gray-600 dark:text-gray-100',
    }

    return (
      <div
        ref={ref}
        className={cn(
          'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
          variants[variant],
          className,
        )}
        {...props}
      />
    )
  },
)

Badge.displayName = 'Badge'

export { Badge }

import React from 'react'

export interface SelectProps
  extends React.SelectHTMLAttributes<HTMLSelectElement> {
  children: React.ReactNode
}

export interface SelectTriggerProps {
  children: React.ReactNode
  className?: string
}

export interface SelectContentProps {
  children: React.ReactNode
}

export interface SelectItemProps
  extends React.OptionHTMLAttributes<HTMLOptionElement> {
  children: React.ReactNode
}

export interface SelectValueProps {
  placeholder?: string
}

const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  ({ className, children, ...props }, ref) => (
    <select
      className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${
        className || ''
      }`}
      ref={ref}
      {...props}
    >
      {children}
    </select>
  ),
)
Select.displayName = 'Select'

// For compatibility with shadcn/ui patterns, we'll make these work with native select
const SelectTrigger: React.FC<SelectTriggerProps> = ({ children }) => (
  <>{children}</>
)
const SelectContent: React.FC<SelectContentProps> = ({ children }) => (
  <>{children}</>
)
const SelectValue: React.FC<SelectValueProps> = () => null // Not needed for native select

const SelectItem = React.forwardRef<HTMLOptionElement, SelectItemProps>(
  ({ children, ...props }, ref) => (
    <option ref={ref} {...props}>
      {children}
    </option>
  ),
)
SelectItem.displayName = 'SelectItem'

export {
  Select, SelectTrigger, SelectContent, SelectItem, SelectValue,
}

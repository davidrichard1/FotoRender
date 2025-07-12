import React from 'react'

interface AccordionProps {
  title: string
  subtitle?: string
  isOpen: boolean
  onToggle: () => void
  children: React.ReactNode
  actions?: React.ReactNode
}

export function Accordion({
  title,
  subtitle,
  isOpen,
  onToggle,
  children,
  actions,
}: AccordionProps) {
  return (
    <div className="bg-[#0D1117] rounded border border-[#30363D]">
      <button
        onClick={onToggle}
        className="accordion-header w-full flex items-center justify-between p-4 text-left hover:bg-[#161B22] transition-colors"
      >
        <div className="flex flex-col pointer-events-none">
          <h3 className="text-sm font-medium text-white">{title}</h3>
          {subtitle && !isOpen && (
            <p className="text-xs text-gray-400 mt-1">{subtitle}</p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {actions && <div onClick={(e) => e.stopPropagation()}>{actions}</div>}
          <svg
            className={`w-4 h-4 text-gray-400 transition-transform ${
              isOpen ? 'rotate-180' : ''
            }`}
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
        </div>
      </button>

      <div
        className="accordion-content px-4 pb-4"
        data-collapsed={!isOpen}
        style={{
          display: isOpen ? 'block' : 'none',
        }}
      >
        {children}
      </div>
    </div>
  )
}

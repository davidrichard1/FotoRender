import React, { useState } from 'react'
import { Button } from './button'
import { Input } from './input'
import { Label } from './label'

interface ListManagerProps {
  label: string
  value: string[]
  onChange: (newValue: string[]) => void
  placeholder?: string
  addButtonText?: string
  className?: string
  itemClassName?: string
}

export function ListManager({
  label,
  value,
  onChange,
  placeholder = "Enter new item...",
  addButtonText = "Add",
  className = "",
  itemClassName = ""
}: ListManagerProps) {
  const [newItem, setNewItem] = useState("")
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editingValue, setEditingValue] = useState("")

  const handleAddItem = () => {
    if (newItem.trim() && !value.includes(newItem.trim())) {
      onChange([...value, newItem.trim()])
      setNewItem("")
    }
  }

  const handleRemoveItem = (index: number) => {
    const newValue = value.filter((_, i) => i !== index)
    onChange(newValue)
  }

  const handleMoveUp = (index: number) => {
    if (index > 0) {
      const newValue = [...value]
      const temp = newValue[index]
      newValue[index] = newValue[index - 1]
      newValue[index - 1] = temp
      onChange(newValue)
    }
  }

  const handleMoveDown = (index: number) => {
    if (index < value.length - 1) {
      const newValue = [...value]
      const temp = newValue[index]
      newValue[index] = newValue[index + 1]
      newValue[index + 1] = temp
      onChange(newValue)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddItem()
    }
  }

  const handleStartEdit = (index: number) => {
    setEditingIndex(index)
    setEditingValue(value[index])
  }

  const handleSaveEdit = () => {
    if (editingIndex !== null && editingValue.trim()) {
      const newValue = [...value]
      newValue[editingIndex] = editingValue.trim()
      onChange(newValue)
      setEditingIndex(null)
      setEditingValue("")
    }
  }

  const handleCancelEdit = () => {
    setEditingIndex(null)
    setEditingValue("")
  }

  const handleEditKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSaveEdit()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      handleCancelEdit()
    }
  }

  return (
    <div className={`space-y-3 ${className}`}>
      <Label className="text-sm font-medium">{label}</Label>
      
      {/* Existing items - contained in a bordered section */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg bg-gray-50/30 dark:bg-gray-800/30 p-3">
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {value.map((item, index) => (
            <div
              key={index}
              className={`flex items-start gap-2 p-3 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md group hover:bg-gray-50 dark:hover:bg-gray-800 hover:border-gray-300 dark:hover:border-gray-600 transition-all shadow-sm ${itemClassName}`}
            >
              {editingIndex === index ? (
                // Edit mode
                <div className="flex-1 space-y-2">
                  <textarea
                    value={editingValue}
                    onChange={(e) => setEditingValue(e.target.value)}
                    onKeyDown={handleEditKeyPress}
                    className="w-full text-sm p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 resize-none min-h-[2.5rem]"
                    autoFocus
                    rows={Math.min(Math.max(Math.ceil(editingValue.length / 60), 1), 4)}
                  />
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      size="sm"
                      onClick={handleSaveEdit}
                      className="px-3 py-1 text-xs"
                    >
                      Save
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={handleCancelEdit}
                      className="px-3 py-1 text-xs"
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              ) : (
                // Display mode
                <>
                  <div className="flex-1 min-w-0">
                    <p 
                      className="text-sm break-words cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                      onClick={() => handleStartEdit(index)}
                      title="Click to edit"
                    >
                      {item}
                    </p>
                    {item.length > 100 && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {item.length} characters
                      </p>
                    )}
                  </div>
                  
                  {/* Action buttons */}
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleStartEdit(index)}
                      className="h-6 w-6 p-0 hover:bg-green-100 dark:hover:bg-green-900"
                      title="Edit item"
                    >
                      ✏️
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleMoveUp(index)}
                      disabled={index === 0}
                      className="h-6 w-6 p-0 hover:bg-blue-100 dark:hover:bg-blue-900"
                      title="Move up"
                    >
                      ↑
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleMoveDown(index)}
                      disabled={index === value.length - 1}
                      className="h-6 w-6 p-0 hover:bg-blue-100 dark:hover:bg-blue-900"
                      title="Move down"
                    >
                      ↓
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveItem(index)}
                      className="h-6 w-6 p-0 text-red-500 hover:text-red-700 hover:bg-red-100 dark:hover:bg-red-900"
                      title="Remove item"
                    >
                      ×
                    </Button>
                  </div>
                </>
              )}
            </div>
          ))}
          
          {value.length === 0 && (
            <div className="text-sm text-gray-500 dark:text-gray-400 italic p-3 text-center border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-800">
              No items added yet
            </div>
          )}
        </div>
        
        {/* Add new item - inside the same container */}
        <div className="space-y-2 mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
          <textarea
            value={newItem}
            onChange={(e) => setNewItem(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleAddItem()
              }
            }}
            placeholder={placeholder}
            className="w-full text-sm p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 resize-none min-h-[2.5rem]"
            rows={Math.min(Math.max(Math.ceil(newItem.length / 60), 1), 3)}
          />
          <div className="flex gap-2">
            <Button
              type="button"
              onClick={handleAddItem}
              disabled={!newItem.trim() || value.includes(newItem.trim())}
              className="px-4"
            >
              {addButtonText}
            </Button>
            {newItem.trim() && (
              <Button
                type="button"
                variant="outline"
                onClick={() => setNewItem("")}
                className="px-3"
              >
                Clear
              </Button>
            )}
          </div>
        </div>
        
        {/* Item count */}
        {value.length > 0 && (
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-right">
            {value.length} item{value.length !== 1 ? 's' : ''}
          </div>
        )}
      </div>
    </div>
  )
} 
'use client'

import React, { useState, useMemo, useEffect } from 'react'
import {
  useReactTable,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  flexRender,
  ColumnDef,
  SortingState,
  ColumnFiltersState,
  VisibilityState,
  PaginationState,
} from '@tanstack/react-table'
import {
  Button, Card, CardContent, Badge, Input,
} from '@/components/ui'
import { fetchModels, ApiModel, refreshData } from '@/lib/api'

interface AiModel {
  id: string
  filename: string
  displayName: string
  description?: string
  modelType: 'SDXL' | 'SD15' | 'FLUX' | 'OTHER'
  baseModel: string
  isNsfw: boolean
  isActive: boolean
  author?: string
  version?: string
  tags: string[]
  usageCount: number
  lastUsed?: string
  createdAt: string
  updatedAt: string
}

// Convert API model to UI model format
const convertApiModelToUiModel = (
  apiModel: ApiModel,
  index: number,
): AiModel => {
  // Parse base model and convert to display format
  const baseModelMap: Record<string, string> = {
    noobai: 'NOOBAI',
    ponyxl: 'PONYXL',
    illustrious: 'ILLUSTRIOUS',
    'sdxl-base': 'SDXL_BASE',
    sdxl_base: 'SDXL_BASE',
  }

  const baseModel = baseModelMap[apiModel.base_model] || 'OTHER'

  return {
    id: `${index + 1}`,
    filename: apiModel.filename,
    displayName: apiModel.display_name,
    description: apiModel.description || undefined,
    modelType:
      (apiModel.type?.toUpperCase() as 'SDXL' | 'SD15' | 'FLUX' | 'OTHER')
      || 'SDXL',
    baseModel,
    isNsfw: apiModel.is_nsfw,
    isActive: true, // Assume all models from API are active
    author: 'Unknown', // API doesn't provide author info yet
    version: undefined,
    tags: [], // API doesn't provide tags yet
    usageCount: apiModel.usage_count,
    lastUsed: apiModel.last_used || undefined,
    createdAt: new Date().toISOString(), // API doesn't provide created_at yet
    updatedAt: new Date().toISOString(), // API doesn't provide updated_at yet
  }
}

interface ModelsTableProps {
  onAdd: () => void
  onEdit: (model: AiModel) => void
  onDelete: (model: AiModel) => void
}

export const ModelsTable: React.FC<ModelsTableProps> = ({
  onAdd,
  onEdit,
  onDelete,
}) => {
  const [data, setData] = useState<AiModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState({})
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 10,
  })

  // Function to load models (with smart caching)
  const loadModels = async (forceRefresh = false) => {
    try {
      setLoading(true)
      setError(null)

      // Force refresh cache if requested
      if (forceRefresh) {
        refreshData('/models')
      }

      const response = await fetchModels()

      if (response.error) {
        setError(response.error)
        setData([])
      } else if (response.data) {
        const convertedModels = response.data.models.map(
          convertApiModelToUiModel,
        )
        setData(convertedModels)
        console.log('📦 Loaded models from cache/API:', convertedModels.length)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
      setData([])
    } finally {
      setLoading(false)
    }
  }

  // Fetch models on component mount (uses cache if available)
  useEffect(() => {
    loadModels()
  }, [])

  const columns = useMemo<ColumnDef<AiModel>[]>(
    () => [
      {
        accessorKey: 'displayName',
        header: 'Model',
        cell: ({ row }) => (
          <div className="space-y-1">
            <div className="font-medium text-gray-900 dark:text-white">
              {row.original.displayName}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 font-mono">
              {row.original.filename}
            </div>
          </div>
        ),
      },
      {
        accessorKey: 'baseModel',
        header: 'Base Model',
        cell: ({ getValue }) => (
          <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
            {getValue() as string}
          </Badge>
        ),
      },
      {
        accessorKey: 'modelType',
        header: 'Type',
        cell: ({ getValue }) => (
          <Badge className="bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
            {getValue() as string}
          </Badge>
        ),
      },
      {
        accessorKey: 'isNsfw',
        header: 'NSFW',
        cell: ({ getValue }) => (
          <Badge
            className={
              (getValue() as boolean)
                ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
            }
          >
            {(getValue() as boolean) ? 'Yes' : 'No'}
          </Badge>
        ),
      },
      {
        accessorKey: 'isActive',
        header: 'Status',
        cell: ({ getValue }) => (
          <Badge
            className={
              (getValue() as boolean)
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
            }
          >
            {(getValue() as boolean) ? 'Active' : 'Inactive'}
          </Badge>
        ),
      },
      {
        accessorKey: 'usageCount',
        header: 'Usage',
        cell: ({ getValue }) => (
          <span className="text-gray-900 dark:text-white font-medium">
            {(getValue() as number).toLocaleString()}
          </span>
        ),
      },
      {
        accessorKey: 'author',
        header: 'Author',
        cell: ({ getValue }) => (
          <span className="text-gray-600 dark:text-gray-400">
            {(getValue() as string) || 'Unknown'}
          </span>
        ),
      },
      {
        id: 'actions',
        header: 'Actions',
        cell: ({ row }) => (
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => onEdit(row.original)}
              className="h-8 px-3 text-xs"
            >
              Edit
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => onDelete(row.original)}
              className="h-8 px-3 text-xs text-red-600 hover:text-red-700 dark:text-red-400"
            >
              Delete
            </Button>
          </div>
        ),
      },
    ],
    [onEdit, onDelete],
  )

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    onPaginationChange: setPagination,
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
      pagination,
    },
  })

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h2 className="text-2xl font-semibold tracking-tight text-gray-900 dark:text-white">
              AI Models
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Loading models...
            </p>
          </div>
        </div>

        <Card>
          <CardContent className="p-6">
            <div className="animate-pulse space-y-4">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4"></div>
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className="h-12 bg-gray-200 dark:bg-gray-700 rounded"
                  ></div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h2 className="text-2xl font-semibold tracking-tight text-gray-900 dark:text-white">
              AI Models
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Failed to load models
            </p>
          </div>
        </div>

        <Card>
          <CardContent className="p-6">
            <div className="text-center space-y-4">
              <div className="text-red-600 dark:text-red-400">
                <p className="font-medium">Error loading models</p>
                <p className="text-sm mt-1">{error}</p>
              </div>
              <Button
                onClick={() => window.location.reload()}
                variant="outline"
              >
                Retry
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h2 className="text-2xl font-semibold tracking-tight text-gray-900 dark:text-white">
            AI Models
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Manage your AI models and their configurations
          </p>
        </div>
        <Button onClick={onAdd} className="bg-blue-600 hover:bg-blue-700">
          Add Model
        </Button>
      </div>

      <Card>
        <CardContent className="space-y-4">
          <div className="flex items-center space-x-2">
            <Input
              placeholder="Filter models..."
              value={
                (table.getColumn('displayName')?.getFilterValue() as string)
                ?? ''
              }
              onChange={(event) => table
                .getColumn('displayName')
                ?.setFilterValue(event.target.value)
              }
              className="max-w-sm"
            />
            <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400">
              <span>Show:</span>
              <select
                value={table.getState().pagination.pageSize}
                onChange={(e) => {
                  table.setPageSize(Number(e.target.value))
                }}
                className="border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-800"
              >
                {[10, 20, 30, 40, 50].map((pageSize) => (
                  <option key={pageSize} value={pageSize}>
                    {pageSize}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="rounded-md border border-gray-200 dark:border-gray-700">
            <table className="w-full">
              <thead>
                {table.getHeaderGroups().map((headerGroup) => (
                  <tr
                    key={headerGroup.id}
                    className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800"
                  >
                    {headerGroup.headers.map((header) => (
                      <th
                        key={header.id}
                        className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                      >
                        {header.isPlaceholder
                          ? null
                          : flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                      </th>
                    ))}
                  </tr>
                ))}
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row) => (
                    <tr
                      key={row.id}
                      className="hover:bg-gray-50 dark:hover:bg-gray-800"
                    >
                      {row.getVisibleCells().map((cell) => (
                        <td
                          key={cell.id}
                          className="px-4 py-4 whitespace-nowrap text-sm"
                        >
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext(),
                          )}
                        </td>
                      ))}
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td
                      colSpan={columns.length}
                      className="px-4 py-8 text-center text-gray-500 dark:text-gray-400"
                    >
                      No models found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between px-2">
            <div className="flex-1 text-sm text-gray-600 dark:text-gray-400">
              Showing{' '}
              {table.getState().pagination.pageIndex
                * table.getState().pagination.pageSize
                + 1}{' '}
              to{' '}
              {Math.min(
                (table.getState().pagination.pageIndex + 1)
                  * table.getState().pagination.pageSize,
                table.getFilteredRowModel().rows.length,
              )}{' '}
              of {table.getFilteredRowModel().rows.length} models
            </div>
            <div className="flex items-center space-x-6 lg:space-x-8">
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  className="h-8 w-8 p-0"
                  onClick={() => table.setPageIndex(0)}
                  disabled={!table.getCanPreviousPage()}
                >
                  {'<<'}
                </Button>
                <Button
                  variant="outline"
                  className="h-8 w-8 p-0"
                  onClick={() => table.previousPage()}
                  disabled={!table.getCanPreviousPage()}
                >
                  {'<'}
                </Button>
                <Button
                  variant="outline"
                  className="h-8 w-8 p-0"
                  onClick={() => table.nextPage()}
                  disabled={!table.getCanNextPage()}
                >
                  {'>'}
                </Button>
                <Button
                  variant="outline"
                  className="h-8 w-8 p-0"
                  onClick={() => table.setPageIndex(table.getPageCount() - 1)}
                  disabled={!table.getCanNextPage()}
                >
                  {'>>'}
                </Button>
              </div>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                Page {table.getState().pagination.pageIndex + 1} of{' '}
                {table.getPageCount()}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

import { dirname } from 'path'
import { fileURLToPath } from 'url'
import { FlatCompat } from '@eslint/eslintrc'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const compat = new FlatCompat({
  baseDirectory: __dirname
})

const eslintConfig = [
  {
    ignores: [
      '**/node_modules/**',
      '**/.next/**',
      '**/dist/**',
      '**/build/**',
      '**/generated/**',
      '**/src/generated/**'
    ]
  },
  ...compat.extends('next/core-web-vitals', 'next/typescript'),
  {
    files: ['**/*.{js,mjs,cjs,jsx,ts,tsx}'],
    rules: {
      // Essential code quality rules (keep these)
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
      'prefer-const': 'off',
      'no-var': 'error',
      'no-console': 'off', // Allow console during development

      // Modern JavaScript/TypeScript patterns (helpful)
      'object-shorthand': 'error',
      'prefer-template': 'off',
      'prefer-arrow-callback': 'error',
      'arrow-body-style': 'off',

      // Import/export rules (useful for organization)
      'import/prefer-default-export': 'off', // Allow named exports
      'import/no-unresolved': 'off', // Next.js handles this
      'import/extensions': 'off', // TypeScript handles this
      'import/no-extraneous-dependencies': 'off', // Too restrictive

      // React-specific rules
      'react/react-in-jsx-scope': 'off', // Not needed in React 17+
      'react/prop-types': 'off', // Using TypeScript for prop validation
      'jsx-a11y/anchor-is-valid': 'off', // Next.js Link component
      'react/jsx-filename-extension': [
        'error',
        { extensions: ['.jsx', '.tsx'] }
      ],

      // Remove annoying formatting rules
      indent: 'off', // Let prettier handle this
      'linebreak-style': 'off', // Windows compatibility
      semi: 'off', // Let prettier handle this
      quotes: 'off', // Let prettier handle this
      'comma-dangle': 'off', // Let prettier handle this
      'object-curly-spacing': 'off', // Let prettier handle this
      'array-bracket-spacing': 'off', // Let prettier handle this
      'space-before-blocks': 'off', // Let prettier handle this
      'keyword-spacing': 'off', // Let prettier handle this
      'space-infix-ops': 'off', // Let prettier handle this
      'eol-last': 'off', // Let prettier handle this
      'no-trailing-spaces': 'off', // Let prettier handle this
      'max-len': 'off', // Too restrictive for modern development

      // Relaxed rules for practical development
      'no-underscore-dangle': 'off', // Sometimes needed for Next.js and APIs
      camelcase: 'off', // Too restrictive for API integration
      'no-use-before-define': 'off', // React components can reference each other
      '@typescript-eslint/no-use-before-define': 'off', // Sometimes necessary

      // Keep security-related rules
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-new-func': 'error',
      'no-script-url': 'error',

      // Keep potential bug-catching rules
      'no-unreachable': 'error',
      'no-duplicate-case': 'error',
      'no-empty': 'error',
      'no-irregular-whitespace': 'error',
      'valid-typeof': 'error',

      // Disable strict TypeScript rules temporarily
      '@typescript-eslint/no-explicit-any': 'off',
      'react-hooks/exhaustive-deps': 'off',
      '@next/next/no-img-element': 'off',
      '@next/next/no-async-client-component': 'off'
    }
  }
]

export default eslintConfig

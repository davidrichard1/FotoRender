import { useEffect, useCallback } from 'react'

export function useScrollSync() {
  const initializeScrollSync = useCallback(() => {
    if (typeof window === 'undefined') return

    let frameCount = 0
    let lastTime = performance.now()
    let isMonitoring = false
    let leftContainer: HTMLElement | null = null
    let rightContainer: HTMLElement | null = null
    let momentumSyncActive = false

    const measureScrollFPS = () => {
      frameCount++
      const currentTime = performance.now()

      if (currentTime - lastTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime))
        if (process.env.NODE_ENV === 'development') {
          console.log(`🚀 Synchronized Scroll FPS: ${fps}`)
        }
        frameCount = 0
        lastTime = currentTime
      }

      if (isMonitoring) {
        requestAnimationFrame(measureScrollFPS)
      }
    }

    const synchronizeScrollMomentum = () => {
      leftContainer = document.querySelector('.scroll-sync-left')
      rightContainer = document.querySelector('.scroll-sync-right')

      if (!leftContainer || !rightContainer) return

      const leftMomentum = { velocity: 0, timestamp: 0, position: 0 }
      const rightMomentum = { velocity: 0, timestamp: 0, position: 0 }
      let syncTimeout: NodeJS.Timeout

      const calculateMomentum = (
        container: HTMLElement,
        momentum: typeof leftMomentum
      ) => {
        const currentTime = performance.now()
        const currentPosition = container.scrollTop
        const timeDelta = currentTime - momentum.timestamp

        if (timeDelta > 0) {
          const positionDelta = currentPosition - momentum.position
          momentum.velocity = positionDelta / timeDelta
          momentum.timestamp = currentTime
          momentum.position = currentPosition
        }
      }

      const createSyncHandler = (
        sourceContainer: HTMLElement,
        sourceMomentum: typeof leftMomentum
      ) => {
        let scrolling = false

        return () => {
          if (!momentumSyncActive) return

          calculateMomentum(sourceContainer, sourceMomentum)

          if (!scrolling) {
            scrolling = true
            isMonitoring = true
            measureScrollFPS()

            sourceContainer.style.willChange =
              'scroll-position, transform, contents'
            sourceContainer.style.transform = 'translate3d(0, 0, 0) scale(1)'

            if (leftContainer && rightContainer) {
              leftContainer.style.willChange =
                'scroll-position, transform, contents'
              rightContainer.style.willChange =
                'scroll-position, transform, contents'
              leftContainer.style.transform = 'translate3d(0, 0, 0) scale(1)'
              rightContainer.style.transform = 'translate3d(0, 0, 0) scale(1)'
            }
          }

          clearTimeout(syncTimeout)
          syncTimeout = setTimeout(() => {
            scrolling = false
            isMonitoring = false

            setTimeout(() => {
              if (leftContainer && rightContainer) {
                leftContainer.style.willChange = 'auto'
                rightContainer.style.willChange = 'auto'
              }
            }, 200)
          }, 100)
        }
      }

      leftMomentum.timestamp = performance.now()
      rightMomentum.timestamp = performance.now()
      leftMomentum.position = leftContainer.scrollTop
      rightMomentum.position = rightContainer.scrollTop

      const leftScrollHandler = createSyncHandler(leftContainer, leftMomentum)
      const rightScrollHandler = createSyncHandler(
        rightContainer,
        rightMomentum
      )

      leftContainer.addEventListener('scroll', leftScrollHandler, {
        passive: true
      })
      rightContainer.addEventListener('scroll', rightScrollHandler, {
        passive: true
      })

      momentumSyncActive = true

      return () => {
        momentumSyncActive = false
        if (leftContainer && rightContainer) {
          leftContainer.removeEventListener('scroll', leftScrollHandler)
          rightContainer.removeEventListener('scroll', rightScrollHandler)
        }
        clearTimeout(syncTimeout)
      }
    }

    const applyPlatformMomentum = () => {
      const userAgent = navigator.userAgent.toLowerCase()
      let platformClass = 'momentum-unified'

      if (userAgent.includes('mac') || userAgent.includes('darwin')) {
        platformClass = 'macos-momentum'
      } else if (userAgent.includes('windows') || userAgent.includes('win')) {
        platformClass = 'windows-momentum'
      } else if (
        userAgent.includes('iphone') ||
        userAgent.includes('ipad') ||
        userAgent.includes('ios')
      ) {
        platformClass = 'ios-momentum'
      }

      setTimeout(() => {
        const containers = document.querySelectorAll('.scroll-sync-container')
        containers.forEach((containerElement) => {
          const container = containerElement as HTMLElement
          container.classList.add(platformClass)
        })

        if (process.env.NODE_ENV === 'development') {
          console.log(`🎯 Applied platform momentum: ${platformClass}`)
        }
      }, 100)
    }

    const applyModernScrollOptimization = () => {
      const containers = document.querySelectorAll('.scroll-sync-container')

      containers.forEach((containerElement) => {
        const container = containerElement as HTMLElement

        const observerOptions = {
          root: container,
          rootMargin: '50px 0px',
          threshold: [0, 0.1, 0.9, 1]
        }

        const boundaryObserver = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            // Note: entry.target available if needed for future enhancements

            if (entry.intersectionRatio <= 0.1) {
              container.classList.add('near-top')
              container.classList.remove('near-bottom')
            } else if (entry.intersectionRatio >= 0.9) {
              container.classList.add('near-bottom')
              container.classList.remove('near-top')
            } else {
              container.classList.remove('near-top', 'near-bottom')
            }
          })
        }, observerOptions)

        const firstChild = container.firstElementChild
        const lastChild = container.lastElementChild

        if (firstChild) boundaryObserver.observe(firstChild)
        if (lastChild && lastChild !== firstChild)
          boundaryObserver.observe(lastChild)

        container.style.scrollBehavior = 'auto'
        container.style.overscrollBehavior = 'contain'

        return () => {
          boundaryObserver.disconnect()
        }
      })
    }

    applyPlatformMomentum()
    applyModernScrollOptimization()
    const momentumCleanup = synchronizeScrollMomentum()

    return () => {
      isMonitoring = false
      momentumSyncActive = false
      if (momentumCleanup) {
        momentumCleanup()
      }
    }
  }, [])

  useEffect(() => {
    const cleanup = initializeScrollSync()
    return cleanup
  }, [initializeScrollSync])
}

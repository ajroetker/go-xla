package pjrt

/*
#include <string.h>
*/
import "C"
import (
	"fmt"
	"math/bits"
	"reflect"
	"runtime"
	"unsafe"

	"github.com/gomlx/go-xla/internal/pool"
)

type arenaContainerBase struct {
	data          *arenaData
	size, current int
	poolIndex     int // index in the arenaPools, -1 if not from pool
	cPointer      *byte
}

// arenaContainer implements a trivial arena object to speed up allocations that will be used in CGO calls.
//
// The issue it is trying to solve is that individual CGO calls are slow, including C.malloc().
//
// It pre-allocates the given size in bytes in C -- so it does not need to be pinned when using CGO and allows
// for fast suballocations.
// It can only be freed all at once.
//
// If you don't call Free at the end, it will leak the C allocated space.
//
// The Plugin object also provides an arenaPool that improves things a bit.
type arenaContainer pool.PoolNode[arenaContainerBase]

// arenaData is an indirection so we can use runtime.AddCleanUp.
type arenaData struct {
	buf []byte
}

// newArena creates a new Arena with the given fixed size.
//
// It provides fast sub-allocations, which can only be freed all at once.
//
// See arenaAlloc, arena.Free and arena.Reset.
func newArena(size int) *arenaContainer {
	buf := cMallocArray[byte](size)
	// fmt.Printf("* arena: alloc(%d)\n", size)
	a := &arenaContainer{
		Item: arenaContainerBase{
			data: &arenaData{
				buf: unsafe.Slice(buf, size),
			},
			size:      size,
			poolIndex: -1,
		},
	}
	runtime.AddCleanup(a, freeArenaData, a.Item.data)
	return a
}

const arenaAlignBytes = 8

// arenaAlloc allocates a type T from the arena. It panics if the arena run out of memory.
func arenaAlloc[T any](a *arenaContainer) (ptr *T) {
	allocSize := cSizeOf[T]()
	if a.Item.current+int(allocSize) > a.Item.size {
		panic(fmt.Sprintf("Arena(%p, size=%d) out of memory while allocating %d bytes for %q", a, a.Item.size, allocSize, reflect.TypeOf(ptr).Elem()))
	}
	ptr = (*T)(unsafe.Pointer(&a.Item.data.buf[a.Item.current]))
	a.Item.current += int(allocSize)
	a.Item.current = (a.Item.current + arenaAlignBytes - 1) &^ (arenaAlignBytes - 1)
	return
}

// arenaAllocSlice allocates an array of n elements of type T from the arena.
//
// It panics if the arena run out of memory.
func arenaAllocSlice[T any](a *arenaContainer, n int) (slice []T) {
	allocSize := C.size_t(n) * cSizeOf[T]()
	if a.Item.current+int(allocSize) > a.Item.size {
		panic(fmt.Sprintf("Arena(%p, size=%d) out of memory while allocating %d bytes for [%d]%s", a, a.Item.size, allocSize, n, reflect.TypeOf(slice).Elem()))
	}
	ptr := (*T)(unsafe.Pointer(&a.Item.data.buf[a.Item.current]))
	a.Item.current += int(allocSize)
	a.Item.current = (a.Item.current + arenaAlignBytes - 1) &^ (arenaAlignBytes - 1)
	slice = unsafe.Slice(ptr, n)
	return
}

// Free invalidates all previous allocations of the arena and frees the C allocated area.
func (a *arenaContainer) Free() {
	if a.Item.data == nil {
		return
	}
	freeArenaData(a.Item.data)
	a.Item.size = 0
	a.Item.current = 0
	a.Item.data = nil
	a.Item.poolIndex = -1
}

func freeArenaData(data *arenaData) {
	if data.buf != nil {
		// fmt.Println("* arena: free")
		cFree(&data.buf[0])
		data.buf = nil
	}
}

// Reset invalidates all previous allocations with the arena, but does not free the C allocated area.
// This way the arena can be re-used.
func (a *arenaContainer) Reset() {
	// Zero the values used.
	if a.Item.data == nil || a.Item.data.buf == nil || a.Item.size == 0 {
		a.Item.current = 0
		return
	}
	if a.Item.current > 0 {
		clearSize := min(a.Item.size, a.Item.current)
		C.memset(unsafe.Pointer(&a.Item.data.buf[0]), 0, C.size_t(clearSize))
	}
	a.Item.current = 0
}

const (
	// minPooledArenaSize is the minimum size for pooled arenas.
	minPooledArenaSize = 2048
	// maxPooledArenaSize is the maximum size for pooled arenas (16MB).
	maxPooledArenaSize = 16 * 1024 * 1024
)

// arenaPools manages pools of arenaContainer objects with power-of-2 sizes.
// It provides fast, concurrent-safe allocation and reuse of arena objects.
//
// It uses internal/pool (and not sync.Pool) because we want to the arenas to live longer.
type arenaPools struct {
	// pools[i] contains arenas of size 2^(i+11), where i=0 is DefaultArenaSize (2048 = 2^11)
	// and the maximum is 16MB (2^24).
	pools []*pool.Pool[arenaContainerBase]

	// minShift is the bit position for DefaultArenaSize (11 for 2048)
	minShift int
	// maxShift is the bit position for maxPooledArenaSize (24 for 16MB)
	maxShift int
}

// newArenaPools creates a new arenaPools manager.
func newArenaPools() (*arenaPools, error) {
	// fmt.Println("* arenaPools: new()")
	minShift := bits.TrailingZeros(uint(minPooledArenaSize))
	maxShift := bits.TrailingZeros(uint(maxPooledArenaSize))
	numPools := maxShift - minShift + 1

	ap := &arenaPools{
		pools:    make([]*pool.Pool[arenaContainerBase], numPools),
		minShift: minShift,
		maxShift: maxShift,
	}

	for poolIdx := range numPools {
		poolSize := 1 << (poolIdx + minShift)
		ap.pools[poolIdx] = pool.New(func() *pool.PoolNode[arenaContainerBase] {
			arena := newArena(poolSize)
			arena.Item.poolIndex = poolIdx
			return (*pool.PoolNode[arenaContainerBase])(arena)
		})
	}
	return ap, nil
}

// Get returns an arenaContainer of at least targetSize bytes.
// The actual size will be the next power-of-2 >= targetSize.
// The returned arena is reset and ready to use.
func (ap *arenaPools) Get(targetSize int) *arenaContainer {
	if targetSize <= minPooledArenaSize {
		targetSize = minPooledArenaSize
	}
	// fmt.Printf("* arenaPools.Get(%d)\n", targetSize)

	// Calculate the next power of 2 >= targetSize
	shift := bits.Len(uint(targetSize - 1))
	if shift < ap.minShift {
		shift = ap.minShift
	}

	// If the requested size is larger than max pooled size, allocate directly
	if shift > ap.maxShift {
		return newArena(targetSize)
	}

	// Calculate pool index and actual size
	poolIndex := shift - ap.minShift

	// Try to get from the pool.
	node := ap.pools[poolIndex].Get()
	return (*arenaContainer)(node)
}

// Return returns an arenaContainer to the pool for reuse.
// The arena is reset before being returned to the pool.
// Arenas larger than maxPooledArenaSize are freed instead of pooled.
func (ap *arenaPools) Return(arena *arenaContainer) {
	// fmt.Printf("* arenaPools.Return(poolIndex=%d)\n", arena.poolIndex)
	if arena == nil || arena.Item.data == nil || arena.Item.data.buf == nil {
		return
	}

	// If not from pool or too large, just free it
	if arena.Item.poolIndex < 0 || arena.Item.poolIndex >= len(ap.pools) {
		arena.Free()
		return
	}

	// Reset and return to the pool.
	arena.Reset()
	ap.pools[arena.Item.poolIndex].Put((*pool.PoolNode[arenaContainerBase])(arena))
}

// Free pools.
func (ap *arenaPools) Free() {
	// internal/pool pools cannot be closed/freed.
	ap.pools = nil
}

<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#sparse-resources">Sparse Resources</a>
<ul class="sectlevel1">
<li><a href="#_binding_sparse_memory">1. Binding Sparse Memory</a></li>
<li><a href="#_sparse_buffers">2. Sparse Buffers</a>
<ul class="sectlevel2">
<li><a href="#_sparse_images">2.1. Sparse Images</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/sparse_resources.html
layout: default
---</p>
</div>
<h1 id="sparse-resources" class="sect0">Sparse Resources</h1>
<div class="paragraph">
<p>Vulkan <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#sparsememory">sparse resources</a> are a way to create <code>VkBuffer</code> and <code>VkImage</code> objects which can be bound non-contiguously to one or more <code>VkDeviceMemory</code> allocations. There are many aspects and features of sparse resources which the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#sparsememory-sparseresourcefeatures">spec</a> does a good job explaining. As the <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#_sparse_resource_implementation_guidelines">implementation guidelines</a> point out, most implementations use sparse resources to expose a linear virtual address range of memory to the application while mapping each sparse block to physical pages when bound.</p>
</div>
<div class="sect1">
<h2 id="_binding_sparse_memory">1. Binding Sparse Memory</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Unlike normal resources that call <code>vkBindBufferMemory()</code> or <code>vkBindImageMemory()</code>, sparse memory is bound via a <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#sparsememory-resource-binding">queue operation</a> <code>vkQueueBindSparse()</code>. The main advantage of this is that an application can rebind memory to a sparse resource throughout its lifetime.</p>
</div>
<div class="paragraph">
<p>It is important to notice that this requires some extra consideration from the application. Applications <strong>must</strong> use synchronization primitives to guarantee that other queues do not access ranges of memory concurrently with a binding change. Also, freeing a <code>VkDeviceMemory</code> object with <code>vkFreeMemory()</code> will <strong>not</strong> cause resources (or resource regions) bound to the memory object to become unbound. Applications must not access resources bound to memory that has been freed.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_sparse_buffers">2. Sparse Buffers</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The following example is used to help visually showcase how a sparse <code>VkBuffer</code> looks in memory. Note, it is not required, but most implementations will use sparse block sizes of 64 KB for <code>VkBuffer</code> (actual size is returned in <code>VkMemoryRequirements::alignment</code>).</p>
</div>
<div class="paragraph">
<p>Imagine a 256 KB <code>VkBuffer</code> where there are 3 parts that an application wants to update separately.</p>
</div>
<div class="ulist">
<ul>
<li>
<p>Section A - 64 KB</p>
</li>
<li>
<p>Section B - 128 KB</p>
</li>
<li>
<p>Section C - 64 KB</p>
</li>
</ul>
</div>
<div class="paragraph">
<p>The following showcases how the application views the <code>VkBuffer</code>:</p>
</div>
<div class="imageblock">
<div class="content">
<img src="images/sparse_resources_buffer.png" alt="sparse_resources_buffer.png">
</div>
</div>
<div class="sect2">
<h3 id="_sparse_images">2.1. Sparse Images</h3>
<div class="sect3">
<h4 id="_mip_tail_regions">2.1.1. Mip Tail Regions</h4>
<div class="paragraph">
<p>Sparse images can be used to update mip levels separately which results in a <a href="https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#sparsememory-miptail">mip tail region</a>. The spec describes the various examples that can occur with diagrams.</p>
</div>
</div>
<div class="sect3">
<h4 id="_basic_sparse_resources_example">2.1.2. Basic Sparse Resources Example</h4>
<div class="paragraph">
<p>The following examples illustrate basic creation of sparse images and binding them to physical memory.</p>
</div>
<div class="paragraph">
<p>This basic example creates a normal <code>VkImage</code> object but uses fine-grained memory allocation to back the resource with multiple memory ranges.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkDevice                device;
VkQueue                 queue;
VkImage                 sparseImage;
VkAllocationCallbacks*  pAllocator = NULL;
VkMemoryRequirements    memoryRequirements = {};
VkDeviceSize            offset = 0;
VkSparseMemoryBind      binds[MAX_CHUNKS] = {}; // MAX_CHUNKS is NOT part of Vulkan
uint32_t                bindCount = 0;

// ...

// Allocate image object
const VkImageCreateInfo sparseImageInfo =
{
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,        // sType
    NULL,                                       // pNext
    VK_IMAGE_CREATE_SPARSE_BINDING_BIT | ...,   // flags
    ...
};
vkCreateImage(device, &amp;sparseImageInfo, pAllocator, &amp;sparseImage);

// Get memory requirements
vkGetImageMemoryRequirements(
    device,
    sparseImage,
    &amp;memoryRequirements);

// Bind memory in fine-grained fashion, find available memory ranges
// from potentially multiple VkDeviceMemory pools.
// (Illustration purposes only, can be optimized for perf)
while (memoryRequirements.size &amp;&amp; bindCount &lt; MAX_CHUNKS)
{
    VkSparseMemoryBind* pBind = &amp;binds[bindCount];
    pBind-&gt;resourceOffset = offset;

    AllocateOrGetMemoryRange(
        device,
        &amp;memoryRequirements,
        &amp;pBind-&gt;memory,
        &amp;pBind-&gt;memoryOffset,
        &amp;pBind-&gt;size);

    // memory ranges must be sized as multiples of the alignment
    assert(IsMultiple(pBind-&gt;size, memoryRequirements.alignment));
    assert(IsMultiple(pBind-&gt;memoryOffset, memoryRequirements.alignment));

    memoryRequirements.size -= pBind-&gt;size;
    offset                  += pBind-&gt;size;
    bindCount++;
}

// Ensure entire image has backing
if (memoryRequirements.size)
{
    // Error condition - too many chunks
}

const VkSparseImageOpaqueMemoryBindInfo opaqueBindInfo =
{
    sparseImage,                                // image
    bindCount,                                  // bindCount
    binds                                       // pBinds
};

const VkBindSparseInfo bindSparseInfo =
{
    VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,         // sType
    NULL,                                       // pNext
    ...
    1,                                          // imageOpaqueBindCount
    &amp;opaqueBindInfo,                            // pImageOpaqueBinds
    ...
};

// vkQueueBindSparse is externally synchronized per queue object.
AcquireQueueOwnership(queue);

// Actually bind memory
vkQueueBindSparse(queue, 1, &amp;bindSparseInfo, VK_NULL_HANDLE);

ReleaseQueueOwnership(queue);</code></pre>
</div>
</div>
</div>
<div class="sect3">
<h4 id="_advanced_sparse_resources">2.1.3. Advanced Sparse Resources</h4>
<div class="paragraph">
<p>This more advanced example creates an arrayed color attachment / texture image and binds only LOD zero and the required metadata to physical memory.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">VkDevice                            device;
VkQueue                             queue;
VkImage                             sparseImage;
VkAllocationCallbacks*              pAllocator = NULL;
VkMemoryRequirements                memoryRequirements = {};
uint32_t                            sparseRequirementsCount = 0;
VkSparseImageMemoryRequirements*    pSparseReqs = NULL;
VkSparseMemoryBind                  binds[MY_IMAGE_ARRAY_SIZE] = {};
VkSparseImageMemoryBind             imageBinds[MY_IMAGE_ARRAY_SIZE] = {};
uint32_t                            bindCount = 0;

// Allocate image object (both renderable and sampleable)
const VkImageCreateInfo sparseImageInfo =
{
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,        // sType
    NULL,                                       // pNext
    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | ..., // flags
    ...
    VK_FORMAT_R8G8B8A8_UNORM,                   // format
    ...
    MY_IMAGE_ARRAY_SIZE,                        // arrayLayers
    ...
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
    VK_IMAGE_USAGE_SAMPLED_BIT,                 // usage
    ...
};
vkCreateImage(device, &amp;sparseImageInfo, pAllocator, &amp;sparseImage);

// Get memory requirements
vkGetImageMemoryRequirements(
    device,
    sparseImage,
    &amp;memoryRequirements);

// Get sparse image aspect properties
vkGetImageSparseMemoryRequirements(
    device,
    sparseImage,
    &amp;sparseRequirementsCount,
    NULL);

pSparseReqs = (VkSparseImageMemoryRequirements*)
    malloc(sparseRequirementsCount * sizeof(VkSparseImageMemoryRequirements));

vkGetImageSparseMemoryRequirements(
    device,
    sparseImage,
    &amp;sparseRequirementsCount,
    pSparseReqs);

// Bind LOD level 0 and any required metadata to memory
for (uint32_t i = 0; i &lt; sparseRequirementsCount; ++i)
{
    if (pSparseReqs[i].formatProperties.aspectMask &amp;
        VK_IMAGE_ASPECT_METADATA_BIT)
    {
        // Metadata must not be combined with other aspects
        assert(pSparseReqs[i].formatProperties.aspectMask ==
               VK_IMAGE_ASPECT_METADATA_BIT);

        if (pSparseReqs[i].formatProperties.flags &amp;
            VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT)
        {
            VkSparseMemoryBind* pBind = &amp;binds[bindCount];
            pBind-&gt;memorySize = pSparseReqs[i].imageMipTailSize;
            bindCount++;

            // ... Allocate memory range

            pBind-&gt;resourceOffset = pSparseReqs[i].imageMipTailOffset;
            pBind-&gt;memoryOffset = /* allocated memoryOffset */;
            pBind-&gt;memory = /* allocated memory */;
            pBind-&gt;flags = VK_SPARSE_MEMORY_BIND_METADATA_BIT;

        }
        else
        {
            // Need a mip tail region per array layer.
            for (uint32_t a = 0; a &lt; sparseImageInfo.arrayLayers; ++a)
            {
                VkSparseMemoryBind* pBind = &amp;binds[bindCount];
                pBind-&gt;memorySize = pSparseReqs[i].imageMipTailSize;
                bindCount++;

                // ... Allocate memory range

                pBind-&gt;resourceOffset = pSparseReqs[i].imageMipTailOffset +
                                        (a * pSparseReqs[i].imageMipTailStride);

                pBind-&gt;memoryOffset = /* allocated memoryOffset */;
                pBind-&gt;memory = /* allocated memory */
                pBind-&gt;flags = VK_SPARSE_MEMORY_BIND_METADATA_BIT;
            }
        }
    }
    else
    {
        // resource data
        VkExtent3D lod0BlockSize =
        {
            AlignedDivide(
                sparseImageInfo.extent.width,
                pSparseReqs[i].formatProperties.imageGranularity.width);
            AlignedDivide(
                sparseImageInfo.extent.height,
                pSparseReqs[i].formatProperties.imageGranularity.height);
            AlignedDivide(
                sparseImageInfo.extent.depth,
                pSparseReqs[i].formatProperties.imageGranularity.depth);
        }
        size_t totalBlocks =
            lod0BlockSize.width *
            lod0BlockSize.height *
            lod0BlockSize.depth;

        // Each block is the same size as the alignment requirement,
        // calculate total memory size for level 0
        VkDeviceSize lod0MemSize = totalBlocks * memoryRequirements.alignment;

        // Allocate memory for each array layer
        for (uint32_t a = 0; a &lt; sparseImageInfo.arrayLayers; ++a)
        {
            // ... Allocate memory range

            VkSparseImageMemoryBind* pBind = &amp;imageBinds[a];
            pBind-&gt;subresource.aspectMask = pSparseReqs[i].formatProperties.aspectMask;
            pBind-&gt;subresource.mipLevel = 0;
            pBind-&gt;subresource.arrayLayer = a;

            pBind-&gt;offset = (VkOffset3D){0, 0, 0};
            pBind-&gt;extent = sparseImageInfo.extent;
            pBind-&gt;memoryOffset = /* allocated memoryOffset */;
            pBind-&gt;memory = /* allocated memory */;
            pBind-&gt;flags = 0;
        }
    }

    free(pSparseReqs);
}

const VkSparseImageOpaqueMemoryBindInfo opaqueBindInfo =
{
    sparseImage,                                // image
    bindCount,                                  // bindCount
    binds                                       // pBinds
};

const VkSparseImageMemoryBindInfo imageBindInfo =
{
    sparseImage,                                // image
    sparseImageInfo.arrayLayers,                // bindCount
    imageBinds                                  // pBinds
};

const VkBindSparseInfo bindSparseInfo =
{
    VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,         // sType
    NULL,                                       // pNext
    ...
    1,                                          // imageOpaqueBindCount
    &amp;opaqueBindInfo,                            // pImageOpaqueBinds
    1,                                          // imageBindCount
    &amp;imageBindInfo,                             // pImageBinds
    ...
};

// vkQueueBindSparse is externally synchronized per queue object.
AcquireQueueOwnership(queue);

// Actually bind memory
vkQueueBindSparse(queue, 1, &amp;bindSparseInfo, VK_NULL_HANDLE);

ReleaseQueueOwnership(queue);</code></pre>
</div>
</div>
</div>
</div>
</div>
</div>
<div id="toc" class="toc">
<div id="toctitle">Table of Contents</div>
<ul class="sectlevel0">
<li><a href="#pnext-and-stype">pNext and sType</a>
<ul class="sectlevel1">
<li><a href="#_two_base_structures">1. Two Base Structures</a></li>
<li><a href="#_setting_pnext_structure_example">2. Setting pNext Structure Example</a></li>
<li><a href="#_reading_pnext_values_example">3. Reading pNext Values Example</a></li>
</ul>
</li>
</ul>
</div>
<hr>
<div class="paragraph">
<p>permalink:/Notes/004-3d-rendering/vulkan/chapters/pnext_and_stype.html
layout: default
---</p>
</div>
<h1 id="pnext-and-stype" class="sect0">pNext and sType</h1>
<div class="paragraph">
<p>People new to Vulkan will start to notice the <code>pNext</code> and <code>sType</code> variables all around the Vulkan Spec. The <code>void* pNext</code> is used to allow for expanding the Vulkan Spec by creating a Linked List between structures. The <code>VkStructureType sType</code> is used by the loader, layers, and implementations to know what type of struct was passed in by <code>pNext</code>. <code>pNext</code> is mostly used when dealing with extensions that expose new structures.</p>
</div>
<div class="sect1">
<h2 id="_two_base_structures">1. Two Base Structures</h2>
<div class="sectionbody">
<div class="paragraph">
<p>The Vulkan API <a href="https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#fundamentals-validusage-pNext">provides two base structures</a>, <code>VkBaseInStructure</code> and <code>VkBaseOutStructure</code>, to be used as a convenient way to iterate through a structure pointer chain.</p>
</div>
<div class="paragraph">
<p>The <code>In</code> of <code>VkBaseInStructure</code> refers to the fact <code>pNext</code> is a <code>const *</code> and are read-only to loader, layers, and driver receiving them. The <code>Out</code> of <code>VkBaseOutStructure</code> refers the <code>pNext</code> being used to return data back to the application.</p>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_setting_pnext_structure_example">2. Setting pNext Structure Example</h2>
<div class="sectionbody">
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">// An example with two simple structures, "a" and "b"
typedef struct VkA {
    VkStructureType sType;
    void* pNext;
    uint32_t value;
} VkA;

typedef struct VkB {
    VkStructureType sType;
    void* pNext;
    uint32_t value;
} VkB;

// A Vulkan Function that takes struct "a" as an argument
// This function is in charge of populating the values
void vkGetValue(VkA* pA);

// Define "a" and "b" and set their sType
struct VkB b = {};
b.sType = VK_STRUCTURE_TYPE_B;

struct VkA a = {};
a.sType = VK_STRUCTURE_TYPE_A;

// Set the pNext pointer from "a" to "b"
a.pNext = (void*)&amp;b;

// Pass "a" to the function
vkGetValue(&amp;a);

// Use the values which were both set from vkGetValue()
printf("VkA value = %u \n", a.value);
printf("VkB value = %u \n", b.value);</code></pre>
</div>
</div>
</div>
</div>
<div class="sect1">
<h2 id="_reading_pnext_values_example">3. Reading pNext Values Example</h2>
<div class="sectionbody">
<div class="paragraph">
<p>Underneath, the loader, layers, and driver are now able to find the chained <code>pNext</code> structures. Here is an example to help illustrate how one <strong>could</strong> implement <code>pNext</code> from the loader, layer, or driver point of view.</p>
</div>
<div class="listingblock">
<div class="content">
<pre class="highlight"><code class="language-cpp" data-lang="cpp">void vkGetValue(VkA* pA) {

    VkBaseOutStructure* next = reinterpret_cast&lt;VkBaseOutStructure*&gt;(pA-&gt;pNext);
    while (next != nullptr) {
        switch (next-&gt;sType) {

            case VK_STRUCTURE_TYPE_B:
                VkB* pB = reinterpret_cast&lt;VkB*&gt;(next);
                // This is where the "b.value" above got set
                pB-&gt;value = 42;
                break;

            case VK_STRUCTURE_TYPE_C:
                // Can chain as many structures as supported
                VkC* pC = reinterpret_cast&lt;VkC*&gt;(next);
                SomeFunction(pC);
                break;

            default:
                LOG("Unsupported sType %d", next-&gt;sType);
        }

        // This works because the first two values of all chainable Vulkan structs
        // are "sType" and "pNext" making the offsets the same for pNext
        next = reinterpret_cast&lt;VkBaseOutStructure*&gt;(next-&gt;pNext);
    }

    // ...
}</code></pre>
</div>
</div>
</div>
</div>
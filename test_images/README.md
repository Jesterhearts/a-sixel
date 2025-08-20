# GENERATED
These generated images are used for comparing different palette quantization and dithering
algorithms. Each one was generated with a different set of instructions intended to stress
different aspects of various algorithms.

# Images and Areas of Interest
- `beach.png`: A mixture image, combining large smooth areas of color with areas of similar color
  but containing a pattern or fine detail. There are also areas of high color variation and
  gradients.

    <details>
    <summary>View Image</summary>
    <img src="beach.png" />
    </details>

    Things to look for:
    - Are the pink dark yellow patterns visible in their respective yellow squares?
    - Does the corral still look colorful?
    - Are the overall yellow/blue dominate tones captured?

- `blob.png`: Very simple gradient blobs. This one isn't typically very interesting unless something
  goes really wrong.

    <details>
    <summary>View Image</summary>
    <img src="blob.png"/>
    </details>

    Things to look for:
    - Are the colors correct?
    - Is the watermark captured?

- `blobs2.png`: Blobs of color around the edges of the LAB color space, with some blobs in the
  middle. Certain encoders (e.g. k-means) can struggle with this image at smaller palette sizes
  because of the way they discover clusters.

    <details>
    <summary>View Image</summary>
    <img src="blobs2.png"/>
    </details>

    Things to look for:
    - Is the mid-grey blob still present?
    - Are the other blobs still well represented?

- `bubbles.png`: This is a mixture of performance test and a test of highly varied colors against a
  dominate background.

    <details>
    <summary>View Image</summary>
    <img src="bubbles.png"/>
    </details>

    Things to look for:
    - Do the bubbles still have their iridescence?
    - Is the background still black?

- `building.png`: A very simple black & white with gradients image. Not very interesting, but sanity
  checks algorithms against an image with an already limited palette.

    <details>
    <summary>View Image</summary>
    <img src="building.png"/>
    </details>

- `buildings.png`: Distinct from the above, this has a highly varied set of flat squares of color
  with a lot of fine details.

    <details>
    <summary>View Image</summary>
    <img src="buildings.png"/>
    </details>

    Things to look for:
    - Do the dominate flat squares still look somewhat correct?
    - Are the fine details still preserved?

- `cherry.png`: This tests a mixture of gradient preservation and highlight extraction.

    <details>
    <summary>View Image</summary>
    <img src="cherry.png"/>
    </details>

    Things to look for:
    - Are the green and red accents still present?
    - Does the cherry still look shiny?
    - Are the trees in the background still distinguishable?
    - Are the highlights on the tree branch still present?

- `clouds.png`: Gradients with lots of detail.
 
    <details>
    <summary>View Image</summary>
    <img src="clouds.png"/>
    </details>

    Things to look for:
    - Is the structure of the clouds still distinguishable?
    - Are the blue and pink highlights still present?

- `confetti1.png`: A simple confetti image with a huge range of lot of colors.

    <details>
    <summary>View Image</summary>
    <img src="confetti1.png"/>
    </details>

    Things to look for:
    - Do the colors look mostly correct?

- `confetti2.png`: Another confetti image, with a large white background near the top.

    <details>
    <summary>View Image</summary>
    <img src="confetti2.png"/>
    </details>

    Things to look for:
    - Do the colors look mostly correct?
    - Is the white background still white?

- `crystal.png`: A mixture of gradients and fine details.
    
    <details>
    <summary>View Image</summary>
    <img src="crystal.png"/>
    </details>

    Things to look for:
    - Are the cracks still clearly distinguishable?

- `crystal2.png`: Another, more colorful crystal image.

    <details>
    <summary>View Image</summary>
    <img src="crystal2.png"/>
    </details>

    Things to look for:
    - Are the rays of light still present/white?
    - Is the background still black?
    - How may of the colors of the facet do there appear to be?

- `feather.png`: A macro shot of a feather with a lot of fine details.

    <details>
    <summary>View Image</summary>
    <img src="feather.png"/>
    </details>

    Things to look for:
    - Are the fine details still present?
    - Are the colors still mostly correct?

- `flat_rainbow.png`: A simple flat rainbow (no gradients/texture as in blobs & confetti).

    <details>
    <summary>View Image</summary>
    <img src="flat_rainbow.png"/>
    </details>

- `flowers.png`: Black and white with blue flower highlights. This tracks how well colorful, fine
  details are extracted for the palette.

    <details>
    <summary>View Image</summary>
    <img src="flowers.png"/>
    </details>

    Things to look for:
    - Are the blue flowers still present & blue?
    - Are the black and white details of the trees still present?
    
- `fog.png`: Fog/forest, a mixture of gradients and details.

    <details>
    <summary>View Image</summary>
    <img src="fog.png"/>
    </details>

    Things to look for:
    - Is the fog gradient still present?
    - Do the mossy rocks still look green?

- `magicalgirl.png`: A small image with lots of detail. This ensures that the algorithm doesn't
  overly "blur" the result image at smaller image sizes.

    <details>
    <summary>View Image</summary>   
    <img src="magicalgirl.png"/>
    </details>

    Things to look for:
    - Are the details still present?
    - Are the colors in e.g. the stars still present?

- `oil.png`: A mixture of gradients with large color blocks. Also slightly tests performance.

    <details>
    <summary>View Image</summary>
    <img src="oil.png"/>
    </details>

    Things to look for:
    - Do the street lights and other lighting details still match the original?
    - Does the sky still look a deep blue?

- `phoenix.png`: A flat image with lots of bold colors. Some algoithms corrupt the bold colors if
  they make bad merge choices, and this image makes it very visually apparent when this happens.

    <details>
    <summary>View Image</summary>
    <img src="phoenix.png"/>
    </details>

    Things to look for:
    - Are the white panels still white?
    - Are the black lines still black?
    - Do the bold primary colors still look correct?

- `rainbows.png`: Pure gradients overlapping.

    <details>
    <summary>View Image</summary>
    <img src="rainbows.png"/>
    </details>

    Things to look for:
    - Have any of the colors degenerated to a brown/grey?

- `room.png`: High variance in lighting.

    <details>   
    <summary>View Image</summary>
    <img src="room.png"/>
    </details>

    Things to look for:
    - Is the window still clearly defined?
    - How about the light rays?
    - Are the dust motes still visible?

- `sample.png`: The test image from hell. Fine gradients, blocks of sold color, fine patterns, low
  contrast text.

    <details>
    <summary>View Image</summary>
    <img src="sample.png"/>
    </details>

    Things to look for:
    - How much of the color wheel is preserved?
    - Is the black & white radar pattern still black & white?
    - Is the text still readable?
    - Do the gradients look reasonable?
    - Is the noise in the grey square still present?

- `scales.png`: Rich colors, gradients, and highlights.

    <details>
    <summary>View Image</summary>
    <img src="scales.png"/>
    </details>

    Things to look for:
    - Do the scales still look shiny?
    - Are the yellow highlights still present?
    - Do the colors still look rich & vibrant?

- `scales2.png`: Another scales image. Iridescent with a big red highlight.

    <details>
    <summary>View Image</summary>
    <img src="scales2.png"/>
    </details>

    Things to look for:
    - Is the red highlight still present and a rich color?
    - Do the scales look iridescent?

- `squares.png`: Large flat squares of color with a lot of subtle texture. This is mostly a color
  matching exercise.

    <details>
    <summary>View Image</summary>
    <img src="squares.png"/>
    </details>

    Things to look for:
    - Are the colors still mostly correct?

- `stars.png`: Mostly black with gradients and a few highlight colored stars.

    <details>
    <summary>View Image</summary>
    <img src="stars.png"/>
    </details>

    Things to look for:
    - Are the stars still visible?
    - Is the nebula gradient still present?
    - Are the highlight stars the right color?

- `threads.png`: Highly textured with rich colors. This stresses color combinations for palette
  extraction.

    <details>
    <summary>View Image</summary>   
    <img src="threads.png"/>
    </details>

    Things to look for:
    - Are the reds still vibrant?
    - Are the darkest patches a dark purple/black?
    - Is the texture of the rug preserved?

- `vector.png`: A simple vector style image with very few, flat colors. More of a sanity check than
  anything else.

    <details>
    <summary>View Image</summary>
    <img src="vector.png"/>
    </details>

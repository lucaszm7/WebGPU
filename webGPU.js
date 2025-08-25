
async function initWebGPU(canvas)
{
    if (!navigator.gpu) 
    {
        throw new Error("WebGPU not supported on this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) 
    {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
        device: device,
        format: canvasFormat,
    });

    return {device, context, canvasFormat};
}

function createRenderPipeline(device, canvasFormat, vertexBufferLayout)
{
    const cellShaderModule = device.createShaderModule({
        label: "Cell shader",
        code: 
        `
            @vertex
            fn vertexMain(@location(0) pos : vec2f) -> @builtin(position) vec4f 
            {
                return vec4f(pos.x, pos.y, 0, 1);
            }

            @fragment
            fn fragmentMain() -> @location(0) vec4f
            {
                return vec4f(1, 0, 0, 1);
            }
        `
    });

    const cellPipeline = device.createRenderPipeline(
        {
            label: "cell_pipeline",
            layout: "auto",
            vertex: 
            {
                module: cellShaderModule,
                entryPoint: "vertexMain",
                buffers: [vertexBufferLayout]
            },
            fragment:
            {
                module: cellShaderModule,
                entryPoint: "fragmentMain",
                targets: 
                [
                    {
                        format: canvasFormat
                    }
                ]
            }
        }
    )

    return cellPipeline;
}

function createVertexBuffer(device)
{
    const vertices = new Float32Array([
        //   X,    Y,
        -0.8, -0.8, // Triangle 1 (Blue)
        0.8, -0.8,
        0.8,  0.8,

        -0.8, -0.8, // Triangle 2 (Red)
        0.8,  0.8,
        -0.8,  0.8,
    ]);

    const vertexBuffer = device.createBuffer({
        label: "cell_vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const vertexBufferLayout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0, // Position, see vertex shader
        }],
    };

    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

    return {vertexBuffer, vertexBufferLayout, vertexLength: vertices.length / 2};
}

async function main()
{
    const canvas = document.querySelector("canvas");

    const { device, context, canvasFormat } = await initWebGPU(canvas);

    const { vertexBuffer, vertexBufferLayout, vertexLength } = createVertexBuffer(device);

    const cellPipeline = createRenderPipeline(device, canvasFormat, vertexBufferLayout);

    const encoder = device.createCommandEncoder();

    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: {r: 0, g: 0, b: 0.4, a: 1},
        }],
    });

    renderPass.setPipeline(cellPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.draw(vertexLength);

    renderPass.end();
    device.queue.submit([encoder.finish()]);
}

main().catch(err => 
{
    console.error(err);
})
let GRID_SIZE_X = 0;
let GRID_SIZE_Y = 0;

const UPDATE_INTERVAL = 50;
const WORKGROUP_SIZE = 8;
const PROPORTION = 8;

function configureCanvas(canvas, device, context) {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio
    ];

    canvas.width = presentationSize[0];
    canvas.height = presentationSize[1];

    GRID_SIZE_X = canvas.width / PROPORTION;
    GRID_SIZE_Y = canvas.height / PROPORTION;

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
        size: presentationSize
    });
}

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

    configureCanvas(canvas, device, context);

    return {device, context, canvasFormat};
}

function createComputeShader(device)
{
    return device.createShaderModule({
        label: "game_of_life_simulation_compute_shader",
        code: 
        `
        @group(0) @binding(0) var<uniform> gridSize: vec2f;
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;
        
        fn cellIndex(cell: vec2u) -> u32
        {
            return (cell.y % u32(gridSize.y)) * u32(gridSize.x) + (cell.x % u32(gridSize.x));
        }
            
        fn cellActive(x: u32, y: u32) -> u32
        {
            return cellStateIn[cellIndex(vec2u(x, y))];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u)
        {
            var activeNeighbors = 0u;
            for (var i = -1; i <= 1; i++)
            {
                for (var j = -1; j <= 1; j++)
                {
                    if (i == 0 && j == 0)
                    {
                        continue;
                    }
                    activeNeighbors += cellActive(cell.x + u32(i), cell.y + u32(j));
                }
            }

            let index = cellIndex(cell.xy);

            switch activeNeighbors {
                case 2: {
                    cellStateOut[index] = cellStateIn[index];
                }
                case 3: {
                    cellStateOut[index] = 1;
                }
                default: {
                    cellStateOut[index] = 0;
                }
            }
        }
        `
    });
}

function createVertexFragmentShader(device)
{
    return device.createShaderModule({
        label: "cell_shader",
        code: 
        `
            struct VertexInput {
                @location(0) pos : vec2f,
                @builtin(instance_index) instance: u32,
            };

            struct VertexOutput {
                @builtin(position) pos: vec4f,
                @location(0) cell: vec2f,
            };

            @group(0) @binding(0) var<uniform> gridSize: vec2f;
            @group(0) @binding(1) var<storage> cellState: array<u32>;

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput 
            {
                let state = f32(cellState[input.instance]);
                let i = f32(input.instance);
                
                let cell = vec2f(i % gridSize.x, floor(i / gridSize.x));

                // 1. Calculate the size of one cell in the -1 to +1 space.
                let cell_size = 2.0 / gridSize;
                
                // 2. Calculate the top-left corner of the cell in the -1 to +1 space.
                let cell_corner = cell * cell_size - 1.0;

                // 3. Calculate the final position of the vertex inside that cell.
                // We flip the Y coordinate to match screen space.
                let final_pos = cell_corner + vec2f(
                    (input.pos.x * state + 1.0) * cell_size.x / 2.0,
                    (input.pos.y * state - 1.0) * cell_size.y / -2.0
                );

                var output: VertexOutput;
                output.pos = vec4f(final_pos, 0.0, 1.0);
                output.cell = cell;
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                let c = input.cell / gridSize;
                return vec4f(c, 1.0 - c.x, 1.0);
            }
        `
    });
}

function createPipelines(device, bindGroupLayout, canvasFormat, vertexBufferLayout)
{
    const cellVertexFragmentShaderModule = createVertexFragmentShader(device);

    const gameOfLifeComputeShader = createComputeShader(device);

    const pipelineLayout = device.createPipelineLayout({
        label: "cell_pipeline_layout",
        bindGroupLayouts: [ bindGroupLayout ],
    });

    const pipelines = 
    [
        device.createRenderPipeline
        ({
            label: "cell_pipeline",
            layout: pipelineLayout,
            vertex: 
            {
                module: cellVertexFragmentShaderModule,
                entryPoint: "vertexMain",
                buffers: [vertexBufferLayout]
            },
            fragment:
            {
                module: cellVertexFragmentShaderModule,
                entryPoint: "fragmentMain",
                targets: 
                [
                    {
                        format: canvasFormat
                    }
                ]
            }
        }),
        device.createComputePipeline
        ({
            label: "simulation_pipeline",
            layout: pipelineLayout,
            compute:
            {
                module: gameOfLifeComputeShader,
                entryPoint: "computeMain",
            }
        })
    ]

    return pipelines;
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

    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/ 0, vertices);

    return {vertexBuffer, vertexBufferLayout, vertexLength: vertices.length / 2};
}

function createUniformBuffer(device)
{
    const uniformArray = new Float32Array([GRID_SIZE_X, GRID_SIZE_Y]);

    const uniformBuffer = device.createBuffer
    ({
        label: "grid_uniform",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(uniformBuffer, /*bufferOffset=*/ 0, uniformArray);

    return uniformBuffer;
}

function createStorageBuffer(device)
{
    const cellStateArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);

    const cellStateStorage = 
    [
        device.createBuffer
        ({
            label: "cell_state_storage_buffer_a",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        device.createBuffer
        ({
            label: "cell_state_storage_buffer_b",
            size: cellStateArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
    ];

    for (let i = 0; i < cellStateArray.length; i++) 
    {
        cellStateArray[i] = Math.random() > 0.5 ? 1 : 0;
    }

    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

    for (let i = 0; i < cellStateArray.length; i++) 
    {
        cellStateArray[i] = 0;
    }

    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArray);

    return cellStateStorage;
}

function createBindGroupLayout(device)
{
    return device.createBindGroupLayout
    ({
        label: "cell_bind_group_layout",
        entries:
        [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                buffer: {} // Grid uniform buffer
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: {type: "read-only-storage"} // Cell state input buffer
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"} // Cell state output buffer
            },
        ]
    });
}

function createBindGroups(device, bindGroupLayout, cellStateBuffer, uniformBuffer)
{
    return [
        device.createBindGroup
        ({
            label: "cell_renderer_bind_group_a",
            layout: bindGroupLayout,
            entries: 
            [{
                binding: 0,
                resource: {buffer: uniformBuffer}
            },
            {
                binding: 1,
                resource: {buffer: cellStateBuffer[0]}
            },
            {
                binding: 2,
                resource: {buffer: cellStateBuffer[1]}
            }]
        }),
        device.createBindGroup
        ({
            label: "cell_renderer_bind_group_b",
            layout: bindGroupLayout,
            entries: 
            [{
                binding: 0,
                resource: {buffer: uniformBuffer}
            },
            {
                binding: 1,
                resource: {buffer: cellStateBuffer[1]}
            },
            {
                binding: 2,
                resource: {buffer: cellStateBuffer[0]}
            }]
        })
    ];
}

async function main()
{
    const canvas = document.querySelector("canvas");

    const { device, context, canvasFormat } = await initWebGPU(canvas);

    const { vertexBuffer, vertexBufferLayout, vertexLength } = createVertexBuffer(device);

    const bindGroupLayout = createBindGroupLayout(device);

    const pipelines = createPipelines(device, bindGroupLayout, canvasFormat, vertexBufferLayout);

    const uniformBuffer = createUniformBuffer(device);

    const cellStateBuffer = createStorageBuffer(device);

    const bindGroups = createBindGroups(device, bindGroupLayout, cellStateBuffer, uniformBuffer);

    setInterval(updateGrid, UPDATE_INTERVAL, context, device, pipelines, vertexBuffer, bindGroups, vertexLength);
}

let STEP = 0;
function updateGrid(context, device, pipelines, vertexBuffer, bindGroups, vertexLength)
{
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(pipelines[1]);
    computePass.setBindGroup(0, bindGroups[STEP % 2]);

    const workGroupCountX = Math.ceil(GRID_SIZE_X / WORKGROUP_SIZE);
    const workGroupCountY = Math.ceil(GRID_SIZE_Y / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workGroupCountX, workGroupCountY);

    computePass.end();

    STEP++;

    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: {r: 0, g: 0, b: 0.4, a: 1},
        }],
    });

    renderPass.setPipeline(pipelines[0]);

    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setBindGroup(0, bindGroups[STEP % 2]);
    renderPass.draw(vertexLength, /*number of instances = */ GRID_SIZE_Y * GRID_SIZE_X);

    renderPass.end();

    device.queue.submit([encoder.finish()]);
}

main().catch(err => 
{
    console.error(err);
})
module ImageWrapper

using ..ExplainableGenAI
using Flux
using Images
using Statistics

export load_image_model, generate_image_with_attribution, compute_gradcam

"""
    load_image_model(model_name::String)

Load a pre-trained image generation model.
"""
function load_image_model(model_name::String)
    # Load a pre-trained image classification model.
    # In practice, this would load a model from Metalhead.jl (ResNet, ViT, etc.).
    if model_name == "simple_cnn"
        # Simple CNN for demonstration
        model = Chain(
            Conv((3, 3), 3=>16, pad=(1,1), relu),
            MaxPool((2,2)),
            Conv((3, 3), 16=>32, pad=(1,1), relu),
            MaxPool((2,2)),
            Conv((3, 3), 32=>64, pad=(1,1), relu),
            GlobalMeanPool(),
            Dense(64, 10)
        )
    else
        println("Warning: Unknown model $model_name. Returning generic CNN.")
        return Flux.Chain(Flux.Conv((3,3), 3=>16, Flux.relu), Flux.GlobalMeanPool(), Flux.Dense(16, 10))
    end
end

"""
    generate_image_with_attribution(model, input_image; target_class=nothing)

Generate an image and capture activations for attribution.
"""
function generate_image_with_attribution(model, input_image; target_class=nothing)
    # Convert image to tensor
    img_tensor = channelview(Float32.(input_image))
    
    # Forward pass with hooks
    activations = Dict()
    
    function hook(name)
        return x -> begin
            activations[name] = x
            return x
        end
    end
    
    # Modify model to capture activations
    hooked_model = Flux.mapchildren(model) do layer
        if layer isa Conv
            return Chain(layer, hook(string(layer)))
        else
            return layer
        end
    end
    
    # Get model output
    output = hooked_model(img_tensor)
    
    # If no target class specified, use the predicted class
    if target_class === nothing
        target_class = argmax(output)
    end
    
    return output, activations, target_class
end

"""
    compute_gradcam(model, input_image, target_class; layer_name="Conv_3")

Compute Grad-CAM attribution for an image model.
"""
function compute_gradcam(model, input_image, target_class; layer_name="Conv_3")
    # Convert image to tensor
    img_tensor = channelview(Float32.(input_image))
    
    # Forward pass to get activations
    output, activations, _ = generate_image_with_attribution(model, input_image; target_class)
    
    # Get target layer activations
    target_activations = activations[layer_name]
    
    # Compute gradients of target class output w.r.t. activations
    grads = gradient(() -> output[target_class], Flux.params(target_activations))[target_activations]
    
    # Global average pooling of gradients
    alpha = mean(grads, dims=(1,2))
    
    # Weighted combination of activations
    gradcam = sum(target_activations .* alpha, dims=3)
    
    # ReLU and normalize
    gradcam = relu.(gradcam)
    gradcam = (gradcam .- minimum(gradcam)) ./ (maximum(gradcam) - minimum(gradcam))
    
    return gradcam
end

end
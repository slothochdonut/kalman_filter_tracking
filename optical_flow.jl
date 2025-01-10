using ImageMagick
using Images
using TestImages
using StaticArrays
using ImageTracking #the package use for optical flow
using ImageView
using LinearAlgebra
using CoordinateTransformations
using Colors
using GLMakie
#Pkg.add("Gtk")
#using Gtk.ShortNames
using VideoIO

include("functions.jl")
file = "data/beetle1.mp4"

#img = read(VideoIO.openvideo("data/beetle1.mp4"))
#img_hsv = HSV.(img)

img_sample = read_video((0, 1), file)[2]
imgs= [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in img_sample]
#=Image Credit:  C. Liu. Beyond Pixels: Exploring New Representations and
#Applications for Motion Analysis. Doctoral Thesis. Massachusetts Institute of
#Technology. May 2009. =#

algorithm = Farneback(10, estimation_window = 5,
                         σ_estimation_window = 1.0,
                         expansion_window = 5,
                         σ_expansion_window = 1.0)

flow = optical_flow(Gray{Float32}.(imgs[1]), Gray{Float32}.(imgs[5]), algorithm)

# flow visualization(hue, saturation, value)
hsv = visualize_flow(flow, ColorBased(), RasterConvention())

channels = channelview(float.(hsv))
channels[3,:,:]

hue_channel = channels[1,:,:]
saturation_channel = channels[2,:,:]
value_channel = channels[3,:,:]

# Create a mask for orange color
# The color actually depends on the direction of movement, so... 
orange_mask = (hue_channel .> 35) .& (hue_channel .< 45) .&  # Hue range for orange
               (saturation_channel .> 0.40) .& (saturation_channel .< 0.60)# Saturation threshold


black_image = zeros(RGB, size(hsv))  # Create an empty RGB image
black_image[orange_mask] .= RGB(1, 0, 0) 
black_image[orange_mask]
black_image

#supposed to be around (768, 310)
#we can convolute the image with a sliding window, assign the mean intensity to each pixel, 
a, b = size(black_image)
values = zeros(RGB, (a, b))
k = 5
for c in CartesianIndices((1:a, 1:b))
    i, j = (c[1], c[2])
    area = intersect(CartesianIndices((i-k:i+k, j-k:j+k)), CartesianIndices((1:a, 1:b)))
    values[i, j] = sum(black_image[area])/length(area)
end
est_loc = findmax(Gray.(values))

#or just use a filter and findmax 
black_image1 = imfilter(black_image, ImageFiltering.Kernel.gaussian(3))
est_loc1 = findmax(Gray.(black_image1))

hsv_resized = imresize(RGB.(hsv), (480, 270))
imshow(hsv_resized)
image(hsv_resized)

save("./optical_flow_farneback_1.png", current_figure())
save("./optical_flow_farneback.jpg", hsv)

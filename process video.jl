using Images
using FileIO
using VideoIO
using GLMakie
const GL = GLMakie
using Colors

f = VideoIO.openvideo("data/beetle1.mp4")
img = read(f)
imgs = [rotr90(Matrix(Gray.(img)))]
i = 1
while !eof(f) && i < 100   #read 100 frames
    global i += 1
    read!(f, img)
    push!(imgs, rotr90(Matrix(Gray.(img))))
end
close(f)

video_length = VideoIO.get_duration("data/beetle1.mp4")
video_frames = VideoIO.get_number_frames("data/beetle1.mp4")
video_fps = video_frames/video_length

imgs1 = imgs

scene, layout = layoutscene(resolution = (1400, 900))

lscenes = layout[1:2, 1:3] = [LScene(scene, camera = cam3d!, raw = false) for _ in 1:6]

[scatter!(lscenes[i], rand(100, 3), color = c)
    for (i, c) in enumerate([:red, :blue, :green, :orange, :black, :gray])]
display(scene)

nlz(x, (a,b) = extrema(x)) = (x .- a)./(b-a)


fig = Figure(resolution = (1200, 900))
#imgs2 = [img.*dim for (img, dim) in zip(imgs, diff(nlz.(imgs)))]
imgs = imgs1
ax = fig[1, 1] = GL.Axis(fig, title = "frames")
sl1 = fig[2, 1] = Slider(fig, range = eachindex(imgs), startvalue = 1)

curimg = lift(i -> imgs[i], sl1.value)
image!(ax, curimg)
fig

scatter!(ax, [1536.0], [620.0], markersize=10, strokecolor=:red, strokewidth=2, color= RGBA(0.2, 1.0, 0.0, 0.0))
reset_limits!(ax) #or ctrl+left_click in the window
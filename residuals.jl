using Distributions, Statistics, LinearAlgebra
using DelimitedFiles
using ImageTransformations
using ImageFiltering
using Colors
using GLMakie; gl = GLMakie
using VideoIO
#using JLD2
include("functions.jl")
file = "data/beetle1.mp4"

#set plot theme
size_theme = Theme(fontsize = 30, resolution = (1500, 800), Axis = (
        backgroundcolor = :gray90,
        xgridcolor = :white,
        ygridcolor = :white,
    ), figsizetitlesize = 10)
set_theme!(size_theme)


Err = readdlm("Err.csv", ',')
Yres = readdlm("Yres.csv", ',')

Path = readdlm("trajectories.csv", ',')
Vel = readdlm("velocity.csv", ',')


fig7 = Figure(resolution = (1500, 1000))
ax1 = fig7[1,1] = gl.Axis(fig7, title="residuals by Kalman filter - x1")
ax2 = fig7[2,1] = gl.Axis(fig7, title="residuals by Kalman filter - x2")
scatter!(ax1, Yres[:,1])
scatter!(ax2, Yres[:,2])
save("figures/residuals_kf.png", fig7)

fig = Figure()
ax = fig[1,1] = gl.Axis(fig, title="Errors by tracking")
scatter!(ax, Err[:])
save("figures/erroroftracking.png", fig)


# make the whole slider of movement/record as animation
intervals = [(0, 10), (14, 98), (102, 117), (118, 129), (131, 148), (149, 162), (164, 167), (171, 190), (190.3, 232), (240, 275)]
N = 0
Ns = []
Imgs = []
intervals = [(102, 117)] #for the frame 5000 - 5500 corresponding to 108-118

for i in intervals
    n, imgs = read_video(i, "$file")
    N += n
    Ns = push!(Ns, n)
    for img in imgs
        Imgs = push!(Imgs, img)
    end
end

fig1 = Figure(resolution=(1500, 1000))
ax1 = fig1[1, 1] = gl.Axis(fig1)
sl1 = fig1[2, 1] = Slider(fig1, range = eachindex(Imgs), startvalue = 400)
curimg = lift(i -> Imgs[i], sl1.value)
image!(ax1, curimg)

let i = 4600
    for n in Ns
        scatter!(ax1, Path[i+1:i+n,:], color= :blue, markersize=5)
        i = i + n
    end
end

curpos = [lift(i -> [Path[i,:][k]], sl1.value) for k in 1:2]
scatter!(ax1, curpos..., color= RGBA(0, 0, 0, 0), markersize=10, strokecolor=:red, strokewidth=3)

curs = [lift(i -> [hcat(Path, Vel)[i,:][k]], sl1.value) for k in 1:4]
arrows!(ax1, curs..., arrowcolor=:red, linecolor=:red, arrowsize=3, linewidth=2, lengthscale=10)
display(fig1)

fig1 = Figure(resolution=(1500, 1000))
ax1 = fig1[1, 1] = gl.Axis(fig1)
image!(ax1, Imgs[1])
let i = 0
    for n in Ns
        lines!(ax1, Path[i+1:i+n,:], color=:orange, alpha=0.5, linestyle=:dash, linewidth=3)
        i = i + n
    end
end


record(fig1, "animation1.mp4") do io
    for i in 1:2500
        image!(ax1, Imgs[i])
        scatter!(ax1, [Path[i,1]],[Path[i,2]], color= RGBA(0, 0, 0, 0), markersize=10, strokecolor=:red, strokewidth=3)
        arrows!(ax1, [Path[i,1]], [Path[i,2]], [Vel[i,1]], [Vel[i,2]], arrowcolor=:red, linecolor=:red, arrowsize=3, linewidth=2, lengthscale=10)
        recordframe!(io)
    end
end

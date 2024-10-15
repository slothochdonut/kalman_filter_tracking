using Distributions, Statistics, LinearAlgebra
using Images
using ImageTransformations
using ImageFiltering
using FileIO
using VideoIO
using Colors
using StaticArrays
using GLMakie; gl = GLMakie
using DelimitedFiles
include("functions.jl")

#file path
file = "beetle1.mp4"

#parameters for kalman filter
I4 = SMatrix{4,4}(1.0I)
I2 = @SMatrix [1.0  0.0
      0.0  1.0]
P0 = I4
F = @SMatrix [
    1.0 0.0 1.0 0.0
    0.0 1.0 0.0 1.0
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
]
Q0 = I4
H = @SMatrix [
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0
]
R = I2

Path = []
Vel = []

#write down useful parts in the video manually to extract data (ignore the parts when the beetle disappear)
intervals = [(0, 10), (14, 98), (102, 117), (118, 129), (131, 148), (149, 162), (164, 167), (171, 190), (190.3, 232), (240, 275)]
for inter in intervals
    n, Imgs = read_video(inter, file)di
    Imgs1 = [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in Imgs]
    point, inner, outer = start_point(Imgs1)
    x0 = SVector(point[1], point[2], 0.0, 0.0)
    #res_l =[]
    let x = x0
        P = P0
        for i in 1:n
            x, P = predict(x, F, P, Q0)
            xobs, err = track(-Imgs1[i], x, 5)
            x, P, yres = correct(x, xobs, P, R, H)
            #push!(res_l, yres)
            push!(Path, SVector(x[1], x[2]))
            push!(Vel, SVector(x[3], x[4]))
        end
    end
end

n, Imgs = read_video((164, 167), file)
Imgs1 = [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in Imgs]
point, inner, outer = start_point(Imgs1)
x0 = SVector(point[1], point[2], 0.0, 0.0)
image(Imgs[1])

img_sample = read_video((0, 1), file)[2]
fig = image(img_sample[1])
scatter!(first.(Path), last.(Path), line = false, markersize = 3, color=:red)
scatter!((point[1], point[2]), markersize=5, color=:red)
burrow = SVector(786, 304)
scatter!(burrow, markersize=30, color=:blue, marker= '▮' )
scatter!(mean(Path), markersize=30, color=:yellow, marker= '⋆' )
current_figure()


#binary plot
image([inner .> 0.08; inner])
maximum(inner)

image(outer .> 0.1)
maximum(outer)

#more important threshold is the inner(low as possible), it decides the intensity of pixel
image((outer .- inner .< 0.06).*(outer .> 0.1) .* (inner .> 0.08))
point = findmax((outer .- inner .< 0.06) .* (outer .> 0.1) .* max.(inner .- 0.08, 0))[2]

writedlm("trajectories.csv", Path, ',')
writedlm("velocity.csv", Vel, ',')


#the data of trajectory and velocity load from csv files
X = readdlm("trajectories.csv", ',')
V = readdlm("velocity.csv", ',')

scatter(X[:, 1][V[:,1] .< 0.0001], X[:, 2][V[:,1] .< 0.0001])

function digging(vel, low_v, duration)
    n = size(vel)[1]
    loc = []
    i = 1
    while i <=n
        count = 0
        if (abs(vel[i]) < low_v) & (i < n)
            j = i+1
            while (abs(vel[j]) < low_v) & (j < n)
                count += 1
                j += 1
            end
            if count >= duration
                push!(loc, (i, j))
                i = j
                print(count)
            else
                i += 1
            end
        else
            i += 1
        end
    end
    return loc
end

idxs = digging(V[:,1], 0.1, 50)


locs= [first.(idxs)[i]:last.(idxs)[i] for i in 1:size(idxs)[1]]
idx2 = last.(x)

fig = Figure(figsize =(900, 600))
ax1 = fig[1, 1] = gl.Axis(fig, title = "location of digging")
[scatter!(ax1, X[i,:], color=:red) for i in locs]
fig


burrow = SVector(786, 304)
ΔX = diff(X, dims =1)
ΔV = diff(V, dims =1)
centered_X = X .- burrow'

X_norm = norm.(eachrow(centered_X))
ΔV_norm = norm.(eachrow(ΔV))
V_norm = norm.(eachrow(V))

scatter(ΔX[:,1])
scatter(ΔX[:,2])
v_pre = mean(V, dims=1)

# regression matrix A
A = hcat(ones(12434), X_norm[2:end], (norm(v_pre).- V_norm[2:end]).*V_norm[2:end])
A1 = hcat(ones(12434), abs.(centered_X[2:end,1]), (v_pre[1] .- abs.(V[2:end,1])).*V[2:end,1])
A2 = hcat(ones(12434), centered_X[2:end,2], (v_pre[2] .- V[2:end,2]).*V[2:end,2])
θ = A\ΔV_norm
θ1 = A1\ΔV[:,1]
θ2 = A2\ΔV[:,2]
θm = norm.(eachrow(ΔX))\ΔV_norm


B = hcat(ones(12434),X_norm[2:end])
α = B\ΔV_norm
fig9 = Figure()
ax1 = fig9[1, 1] = gl.Axis(fig9, title = "regression result- first model")
ax1.xlabel="X(t)-x_burrow"; ax1.ylabel="ΔV(t)"
scatter!(ax1, X_norm[2:end], ΔV_norm)
lines!(ax1, X_norm[2:end], B*α, color=(:red, 0.75), linewidth=2)
fig9

ϵ = ΔV_norm - B*α
e_norm = fit_mle(Normal, ϵ)
fig10 = Figure()
ax1 = fig10[1, 1] = gl.Axis(fig10, title = "residual- first model")
scatter!(ax1, ϵ)
ax2 = fig10[2, 1] = gl.Axis(fig10)
hist!(ax2, ϵ;bins=30, normalization=:pdf, color=:grey, yticks!)
Z = [pdf(e_norm, i) for i in -1:0.01:1]
lines!(ax2, -1:0.01:1, Z, color=(:red, 0.75), linewidth=3)
fig10


#residual
ϵ = ΔV_norm - A*θ
scatter(ϵ)
e_norm = fit_mle(Normal, ϵ)


fig10 = Figure()
ax1 = fig10[1, 1] = gl.Axis(fig10, title = "residual")
scatter!(ax1, ϵ)
ax2 = fig10[2, 1] = gl.Axis(fig10)
hist!(ax2, ϵ;bins=30, normalization=:pdf, color=:grey, yticks!)
Z = [pdf(e_norm, i) for i in -1:0.01:1]
lines!(ax2, -1:0.01:1, Z, color=(:red, 0.75), linewidth=3)
fig10


lines(norm.(eachrow(ΔX))[20*50+1:end])
lines(ΔV_norm[20*50+1:end])
mean(norm.(eachrow(ΔX))[20*50+1:end])


mean(Path)
Vel_matrix = reshape([first.(Vel); last.(Vel)], :, 2)
#Vel_matrix = reshape(reinterpret(Any, Vel), 2, :)
mvnorm = fit(MvNormal, Vel_matrix)

Z = [pdf(mvnorm, [i,j]) for i in -5:0.1:5, j in -5:0.1:5]
heatmap(-5:0.1:5, -5:0.1:5, Z)

#scatter plot of velocity
scatter(Vel_matrix[:,1], Vel_matrix[:,2], color= :red)

scatter(Path_matrix[:,1], Path_matrix[:,2], color= :red)

#do the same for path
Path_matrix = reshape([first.(Path); last.(Path)], :, 2)
mvnorm = fit_mle(MvNormal, Path_matrix)
Z = [pdf(mvnorm, [i,j]) for i in 750:1:850, j in 250:1:400]
heatmap(750:1:850, 250:1:400, Z)
scatter!(first.(Path),last.(Path), linewidth=5, color=:blue)
scatter!(last(Path), markersize=50, color=:red, marker='⋆')
scatter!(first(Path), markersize=50, color=:red, marker='→')
current_figure()
# how to describe the digging behavior? The direction changes more frequently than
# a hypothesis test of Multivariate Normal?

error = [pdf(mvnorm, [i,j]) for [i,j] in Vel_matrix']

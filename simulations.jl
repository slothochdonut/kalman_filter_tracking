using Distributions, Statistics, LinearAlgebra
using GLMakie
using Random
include("functions.jl")

#set plot theme
size_theme = Theme(fontsize = 30, resolution = (1000, 800), Axis = (
        backgroundcolor = :gray90,
        xgridcolor = :white,
        ygridcolor = :white,
    ), figsizetitlesize = 10)
set_theme!(size_theme)

## simulate sample trajectory, random walk
# example 1
Random.seed!(123)
T = 100
x0 = 0.0
s0 = [x0]  #s0 is trajectory without noise
x = x0
for t in 2:T
    x = rand(Normal(x, 1))  #formula 3.1
    push!(s0, x)
end
s = [copy(m) for m in s0]
for t = 2:T
    s[t] = rand(Normal(s[t], 1)) #add a noise? formula 3.2
end

path = [x0] 
P0 = 1.0
Ps = [P0] #the trajectory after using kalman filter
let x = s[1],  #x is observation of each step
    P = P0
    for i = 2:T
        x, P = predict(x, 1.0I, P, 1)  #use the kalman filter function from functions.jl
        obs = s[i]
        x, P = correct(x, obs, P, 1, 1)
        push!(path, x)
        push!(Ps, P)
    end
end
#s0: random walk (we want to learn)
#s: random walk with noise (observations)
#path: best estimate of s0

fig1, ax1, scat1 = lines(s0, color= :pink, linewidth = 5, label="True Location") #real trajectory
band!(1:T, path - sqrt.(Ps), path + sqrt.(Ps))
scatter!(s, markersize = 10, color =:gray, linewidth = 2, label="Measurement")
lines!(path, color =:blue, linewidth = 5, label="Filter Estimate") #observed trajectory
ax1.title = "random walk - 1d"
ax1.xlabel = "time step t"; ax1.ylabel = "x_t"
axislegend(ax1)
fig1

save("figures/random_walk1.png", fig1)

#example2： 2d random walk
Random.seed!(123)
x0 = [0.0, 0.0] #start point

β = [1.0, 0.5]
Q = [1.0 0
    0 1.0]

T = 100

μ = [0.0, 0.0]      #measurement noise
R = [1.0 0.0        #measurement covariance
     0.0 1.0]

F = Matrix{Float64}(I, 2, 2)
H = Matrix{Float64}(I, 2, 2)
P = I

s0, s = sample_trajectory(x0, T, β, Q, μ, R, F, H)

x = s[1]
path = [x]
for i = 2:T
    x, P = predict(x, F, P, Q)
    obs = s[i] #observation from the simulation
    x, P, yres = correct(x, obs, P, R, H)
    push!(path, x)
end

fig2, ax2, scat2 = lines(first.(s0), last.(s0), color = :pink, linewidth = 5, label="True Location") #real trajectory
scatter!(first.(s), last.(s), markersize = 10, color = :gray, label="Measurement")
lines!(first.(path), last.(path), color = :blue, linewidth = 5, label="Filter Estimate") #observed trajectory
ax2.title = "random walk - 2d"
ax2.xlabel = "x_t"; ax2.ylabel = "y_t"
axislegend(ax2, position=:lt)
fig2

save("figures/random_walk2.png", fig2)

# a linear dynamic transition
# parameters for example 3 
F = [1.0 0.0 1.0 0.0
     0.0 1.0 0.0 0.5
     0.0 0.0 1.0 0.0
     0.0 0.0 0.0 1.0]

H = [1.0 0.0 0.0 0.0
     0.0 1.0 0.0 0.0]

x0 = [0.0, 0.0, 0.0, 0.0]
β = [0.0, 0.0, 0.0, 0.0]

Q = Matrix{Float64}(I, 4, 4)

μ = [0.0, 0.0]  
R = [5.0 0.0
     0.0 5.0]  #choose large variance for illustration
P = I
T = 100

s0_1, s_1 = sample_trajectory(x0, T, β, Q, μ, R, F, H)

x = x0
path_1 = [x]
for i = 2:T
    x, P = predict(x, F, P, Q)
    obs = s_1[i]
    x, P, yres = correct(x, obs, P, R, H)
    push!(path_1, x)
end

fig3, ax3, scat3 = lines(first.(s0_1), getindex.(s0_1, 2), color = :pink, linewidth = 5, label="True Location")
scatter!(first.(s_1), getindex.(s_1, 2), markersize = 10, label="Measurement")
lines!(first.(path_1), getindex.(path_1, 2), color =:blue, linewidth = 5, label="Filter Estimate") #observed trajectory
ax3.title= "linear dynamic system"
ax3.xlabel = "x"; ax3.ylabel = "y"
axislegend(ax3)
fig3

save("figures/linear_system.png", fig3)

rmse_filter = sqrt(norm([first.(path_1-s0_1), getindex(path_1-s0_1, 2)]))
rmse_measure= sqrt(norm([first.(s_1)-first.(s0_1), getindex.(s_1, 2)-getindex.(s0_1, 2)]))


###detecting and tracking object
n, m = 500, 500  # no. of pixels
i0 = 50
j0 = 50

# Euclidean distance between (i,j) and (i0, j0)
d((i, j), (i0, j0)) = sqrt((i - i0)^2 + (j - j0)^2)
a = 10^2             # size of the "beetle"

# make a picture (matrix) with a beetle (just a blob)
beetleimg(i0, j0, a) = [exp(-d((i, j), (i0, j0))^2/(2*a)) for i = 1:n, j = 1:m]
#image(beetleimg(100, 100))  a blob at (100, 100)

# Normalize a picture to [0, 1]
nlz(img) = (img .- minimum(img)) / (maximum(img) - minimum(img))
σ = 0.1   # noise on the pictures


fig4 = Figure(resolution = (1500, 750))
ax1 = fig4[1, 1] = Axis(fig4, title = "simulate without noise")
ax2 = fig4[1, 2] = Axis(fig4, title = "simulate with noise")
image!(ax1, beetleimg(100, 100, a))
image!(ax2, nlz(σ*randn(m, n)+ beetleimg(100, 100, a))) #with noise
save("figures/blob.png", fig4)


# make pictures out of positions (image number = T)
imgs = [nlz(σ*randn(m, n) + beetleimg(pos[1]*3+50, pos[2]*3+50, a)) for pos in s0]

using ImageFiltering
imgs_flt = [imfilter(img, ImageFiltering.Kernel.gaussian(5)) for img in imgs]

# find the coordinates of the maximum pixel
myfindmax(img) = convert(Tuple, findmax(img)[2]) #return the index
ys = myfindmax(imgs_flt[1]) #only use the first observation
#ys = [myfindmax(img) for img in imgs]


# tracking the location
est_loc = []
obs = ys
rec = []
for i in 2:T
    est_i, box_i= tracking(obs, imgs_flt[i], 30)
    obs = est_i
    push!(est_loc, est_i)
    push!(rec, box_i)
end

fig4 = Figure(resolution = (500,500))
ax3 = fig4[1, 1] = Axis(fig4, title = "simulate beetle tracking")
sl1 = fig4[2, 1] = Slider(fig4, range = eachindex(imgs), startvalue = 1)
curimg = lift(i -> imgs_flt[i], sl1.value)
image!(ax3, curimg)

line1 = lines!(ax3, first.(s0).*3 .+50, last.(s0).*3 .+50, color = :red, linewidth=3) #original track
line2 = lines!(ax3, first.(est_loc), last.(est_loc), color = :blue, linewidth=3) #observed track
fig4

#=
boximg = lift(i -> poly(recs[i], color = nothing, strokecolor = :red, strokewidth = 5), sl1.value)
Rec(recs[1], color = nothing, strokecolor = :red, strokewidth = 5)
=#

# estimate the "error"
#σest = mean(d.(ys, s)) # average error in image recognition

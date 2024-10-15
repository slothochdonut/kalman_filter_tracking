using Distributions, Statistics, LinearAlgebra
using GLMakie
using Random

#set plot theme
size_theme = Theme(fontsize = 30, resolution=(1000,800), titlesize = 10)
set_theme!(size_theme)

## simulate sample trajectory
#parameters for example 1
x0 = [0.0, 0.0] #start point

β = [1.0, 0.5]
Q = [1.0 0.0
    0.0 1.0]

T = 50

μ = [0.0, 0.0]      #measurement noise
R = [1.0 0.0         #measurement covariance
    0.0 1.0]

F = Matrix{Float64}(I, 2, 2)
H = Matrix{Float64}(I, 2, 2)
P = I

Random.seed!(222)

# bulid trajectory
function sample_trajectory(x, T, β, Q, μ, Σ, F, H)
    s0 = [x]
    for t in 1:T-1
        x = F * x + rand(MultivariateNormal(β, Q))
        push!(s0, x)
    end
    s = [copy(m) for m in s0]
    for t in 2:T
        s[t] = H * s[t] + rand(MultivariateNormal(μ, Σ))
    end
    return (s0, s)
end

s0, s = sample_trajectory(x0, T, β, Q, μ, R, F, H)

fig1, ax1, scat = lines(first.(s0), last.(s0), color = :red, linewidth =5) #real trajectory
scatter!(first.(s), last.(s), markersize = 10)  #observed trajectory
ax1.title = "random walk"
ax1.xlabel = "x"; ax1.ylabel = "y"

# kalman filter
# H: observation matrix
# F: state-trasition
# Q: the covariance of the process noise
# R: the covariance of the observation noise

function predict(x, F, P, Q)
    x = F * x    #transition of state
    P = F * P * F' + Q   #estimate covariance matrix of x
    x, P
end

function correct(x, y, Ppred, R, H)
    yres = y - H * x # innovation residual
    S = (H * Ppred * H' + R) # innovation covariance
    K = Ppred * H' / S # Kalman gain
    x = x + K * yres
    P = (I - K * H) * Ppred * (I - K * H)' + K * R * K' #  Joseph form
    #yres_pos = y - H*x   Measurement post-fit residual
    x, P, yres, S
end

# example 1
x = s[1]
path = [x]
res = []
for i = 2:T
    x, P = predict(x, F, P, Q)
    obs = s[i]
    x, P, yres = correct(x, obs, P, R, H)
    push!(path, x)
    push!(res, yres)
end

lines!(first.(path), last.(path), linestyle = :dash, linewidth =5)  #predict by kalman filter
save("walk1.png", fig1)

#=the sum of variance?
distance1 = sum((hcat(s...) - hcat(s0...)) .^ 2, dims = 1)
distance2 = sum((hcat(path...) - hcat(s0...)) .^ 2, dims = 1)
fig2 = lines(distance1[1, :], color = :red)
lines!(distance2[1, :], color = :blue)
save("residuals.png", fig2)
=#

# a linear dynamic transition
# parameters for example 2
F = [1.0 0.0 1.0 0.0
     0.0 1.0 0.0 1.0
     0.0 0.0 1.0 0.0
     0.0 0.0 0.0 1.0]

H = [1.0 0.0 0.0 0.0
     0.0 1.0 0.0 0.0]

x0 = [0.0, 0.0, 0.0, 0.0]
β = [0.0, 0.0, 0.0, 0.0]
Q = Matrix{Float64}(I, 4, 4)
R = [5.0 0.5
     0.5 5.0]  #choose large variance for illustration
P = I

Random.seed!(321)
s0_1, s_1 = sample_trajectory(x0, T, β, Q, μ, R, F, H)
fig2, ax2, scat2 = lines(first.(s0_1), getindex.(s0_1, 2), color = :red, linewidth=5)
scatter!(first.(s_1), getindex.(s_1, 2), markersize = 10)
ax2.title= "linear dynamic"
ax2.xlabel = "x"; ax2.ylabel = "y"

x = x0
path_1 = [x]
res_1= []
for i = 2:T
    x, P = predict(x, F, P, Q)
    obs = s_1[i]
    x, P, yres = correct(x, obs, P, R, H)
    push!(path_1, x)
    push!(res_1, yres)
end

lines!(first.(path_1), getindex.(path_1,2), linestyle= :dash, linewidth=5)
save("walk2.png", fig2)


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

fig3 = Figure(resolution = (1500,750))
ax1 = fig3[1, 1] = Axis(fig3, title = "simulate without noise")
ax2 = fig3[1, 2] = Axis(fig3, title = "simulate with noise")
image!(ax1, beetleimg(100, 100))
image!(ax2, nlz(σ*randn(m, n)+ beetleimg(100, 100))) #with noise
save("simulation.png", fig3)


# make pictures out of positions (image number = T)
imgs = [nlz(σ*randn(m, n) + beetleimg(pos[1]*3+50, pos[2]*3+50)) for pos in s0]


using ImageFiltering
imgs_flt = [imfilter(img, ImageFiltering.Kernel.gaussian(5)) for img in imgs]

# find the coordinates of the maximum pixel
myfindmax(img) = convert(Tuple, findmax(img)[2]) #return the index
ys = myfindmax(imgs_flt[1]) #only use the first observation
#ys = [myfindmax(img) for img in imgs]

# tracking function
function tracking(obs, img, h)
    i, j = round.(Int, obs)
    c = CartesianIndices((i-h:i+h, j-h:j+h))
    x_est = sum((img[k]*k[1]) for k in c)/sum(img[c])
    y_est = sum((img[k]*k[2]) for k in c)/sum(img[c])
    box = collect(Iterators.product([i-h,i+h], [j-h,j+h]))
    return (x_est, y_est), box
end

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

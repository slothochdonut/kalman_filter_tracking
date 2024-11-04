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

# set 1
Random.seed!(123)
x0 = [0.0, 0.0] #start point

β = [1.0, 1.0]
Q = [1.0 0.0
     0.0 1.0]

#Q1 = [0.9 -0.1; 0.1 0.9]
T = 100

μ = [0.0, 0.0]      #measurement noise
R = [1.0 0.0 
    0.0 1.0]      #measurement covariance
F = Matrix{Float64}(I, 2, 2) 
H = Matrix{Float64}(I, 2, 2)
P = Matrix{Float64}(I, 2, 2)


# set 2 
Random.seed!(234)
F = [1.0 0.0 1.0 0.0
     0.0 1.0 0.0 1.0
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
P = Matrix{Float64}(I, 4, 4)

T = 100


s0, s = sample_trajectory(x0, T, β, Q, μ, R, F, H)

x = x0
path = [x]
Ps = [P]
for i = 2:T
    x, P = predict(x, F, P, Q)  
    obs = s[i] #observation from the simulation
    x, P, yres = correct(x, obs, P, R, H)
    push!(path, x)
    push!(Ps, P)
end

fig2, ax2 = lines(first.(s0), getindex.(s0,2), color = :pink, linewidth = 5, label="True Location") #real trajectory
scatter!(first.(s), getindex.(s,2), markersize = 10, color = :gray, label="Measurement")
lines!(first.(path), getindex.(path,2), color = :blue, linewidth = 5, label="Filtered Estimate") #observed trajectory
ax2.title = "random walk - 2d"
ax2.xlabel = "x_t"; ax2.ylabel = "y_t"
axislegend(ax2, position=:rb)
fig2

#save("originF.png", fig2)


fig4 = Figure(resolution = (1500, 750))
ax1 = fig4[1, 1] = Axis(fig4, title = "when β=[1.0, 0.5], estimate residual of x1, x2 \n calculated by first.(path - s0), last.(path - s0)")
scatter!(ax1, 1:T, first.(path - s0), color = :red, label="residual of x1")
scatter!(ax1, 1:T, getindex.(path - s0, 2), color = :blue, label="residual of x2")
hlines!(ax1, mean(first.(path - s0)), color = :red)
hlines!(ax1, mean(getindex.(path - s0,2)), color = :blue)
axislegend(ax1)
fig4


"""
explore the impact of 
1/ when the start of measurement s[1](ex.[5.0, 5.0]) is far away from x0, the kalman filter
still works when s[2] is measured correctly?
2/ the process noise β not zero, β = (1.0, 0.5)
3/ F not same as the real transition matrix F, ex. F = 0.8*F used in kalman filter
4/ Q1 = [0.9 -0.1; 0.1 0.9]
"""

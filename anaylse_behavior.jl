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
#using ProgressMeter
include("functions.jl")
file = "data/beetle1.mp4"

#set plot theme
size_theme = Theme(fontsize = 30, resolution = (1500, 800), Axis = (
        backgroundcolor = :gray90,
        xgridcolor = :white,
        ygridcolor = :white,
    ), figsizetitlesize = 10)
set_theme!(size_theme)

#parameters for kalman filter
I4 = SMatrix{4,4}(1.0I)
I2 = @SMatrix[1.0  0.0
      0.0  1.0]
P0 = I4
F = @SMatrix [
    1.0 0.0 1.0 0.0
    0.0 1.0 0.0 1.0
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
]
Q = I4
H = @SMatrix [
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0
]
R = I2

#-------Done-------#
#write down useful parts in the video manually to extract data (ignore the parts when the beetle disappear)
intervals = [(0, 10), (14, 98), (102, 117), (118, 129), (131, 148), (149, 162), (164, 167), (171, 190), (190.3, 232), (240, 275)]
intervals = [(0, 14)]
#intervals = [(0, 10), (14, 98), (102, 117)]

Path = Vector{SVector{2, Float64}}()  # Preallocate with specific type
Vel = Vector{SVector{2, Float64}}()
Err = Vector{Float64}()
Yres = Vector{SVector{2, Float64}}()

for inter in intervals
    n, Imgs = read_video(inter, file) # n is the number of imgs for one interval
    Imgs1 = [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in Imgs]
    point, inner, outer = start_point(Imgs1) # it works because the beetle moves when it shows up
    x0 = SVector(point[1], point[2], 0.0, 0.0)
    
    let x = x0
        P = P0
        for i in 1:n
            x, P = predict(x, F, P, Q)
            xobs, err = track(-Imgs1[i], x[1:2], 10)
            x, P, yres = correct(x, xobs, P, R, H)
            push!(Err, err)
            push!(Yres, yres)
            #push!(Path, SVector(x[1], x[2]))
            #push!(Vel, SVector(x[3], x[4]))
        end
    end
end

Yres_max = findmax(norm.(Yres))[2]/50 #the time point of max residual
Err_max = findmax(norm.(Err))[2]/50

f = Figure(resolution=(1200,1000))
ax1 = gl.Axis(f[1,1], xlabel="time(s)", title="Residuals of Kalman Filter, time:0-14 seconds")
scatter!(ax1, 0.02:0.02:14, norm.(Yres))
lines!(ax1, [Yres_max, Yres_max], [0, maximum(norm.(Yres))], linestyle=:dash, color=:red, linewidth=2, label="time = $(Yres_max)s" )
axislegend()
ax2 = gl.Axis(f[2,1], xlabel="time(s)", title="Error of tracking, time:0-14 seconds")
scatter!(ax2, 0.02:0.02:14, norm.(Err))
lines!(ax2, [Err_max, Err_max], [minimum(norm.(Err)), maximum(norm.(Err))], linestyle=:dash, color=:red, linewidth=2, label="time = $(Err_max)s")
axislegend()
save("figures/residuals0-14.png", f)
#writedlm("trajectories.csv", Path, ',')
#writedlm("velocity.csv", Vel, ',')

#writedlm("Err.csv", Err, ',')
#writedlm("Yres.csv", Yres, ',')



#----Done ----#
#the start point we manual choose before is (768.0, 310.0), so the function works well.
img_sample = read_video((13, 14), file)[2]
Path1 = hcat(getindex.(Path, 1), getindex.(Path, 2))
Vel1 = hcat(getindex.(Vel, 1), getindex.(Vel, 2))
f = Figure(resolution=(1500,1000))
ax1 = gl.Axis(f[1,1], title="trajectory of 0-14 seconds")
image!(ax1, img_sample[end])
lines!(ax1, Path1, color=:red, label= "path")
scatter!(ax1, Path1[700,:]', markersize=10, strokecolor=:red, strokewidth=3, color= RGBA(0.2, 1.0, 0.0, 0.0), label="tracked location")
scatter!(ax1, SVector(406, 405), markersize=10, strokecolor=:blue, strokewidth=3, color= RGBA(0.2, 1.0, 0.0, 0.0), label="real location")
axislegend()
save("figures/whereitdoesntwork.png", f)

#if we zoom in the area 
f = Figure(resolution=(1500,1000))
ax2 = gl.Axis(f[1,1], title="trajecotory of 10-14 seconds")
#image!(ax2, img_sample[end])
lines!(ax2, Path1[500:end,:], color=:red, label= "Path")
arrows!(ax2, Path1[500:end,1], Path1[500:end,2], Vel1[500:end,1]*0.5, Vel1[500:end,2]*0.5, color=:green, linewidth=3, label="Velocity")  # Example arrow direction
dung = SVector(343, 419)
scatter!(ax2, dung, markersize=30, color=:blue, marker=Circle, label="Dung pile")
scatter!(ax2, (Path1[end,1], Path1[end,2]), markersize=40, color=:Orange, marker=:star5, label="Stuck location")
axislegend()
save("figures/zoomin.png", f)


"""
curpos = lift(i -> [path3[i]], sl1.value)
scatter!(ax1, curpos, markersize=10, strokecolor=:red, strokewidth=2, color= RGBA(0.2, 1.0, 0.0, 0.0))
image!(Imgs[end])
"""

#----Done ----#
#=
Path = readdlm("trajectories.csv", ',')
Vel = readdlm("velocity.csv", ',')
=#

img_sample = read_video((100, 101), file)[2]
fig = Figure(resolution=(1500, 800))
ax = gl.Axis(fig[1, 1])
image!(ax, img_sample[1])
scatter!(ax, Path[:, 1], Path[:, 2], color = :blue, markersize=3, label="trajectory")
scatter!(ax, Path[1,:]', markersize=10, marker= :circle, color=:pink, label="start point")
burrow = SVector(786, 304)
scatter!(ax, burrow, markersize=30, color=:red, marker=:star5, label="burrow")
axislegend(position=:lt)
#scatter!(mean(Path), markersize=30, color=:yellow, marker= '⋆' )
save("figures/whole.png", fig)



#----Done ----#
#example plot of inner and outer(use first image as example)
img_sample = read_video((0, 1), file)[2]
img_sample1 = [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in img_sample]
point, inner, outer = start_point(img_sample1)

fig8 = Figure(resolution=(1500, 600))
ax1 = fig8[1, 1] = gl.Axis(fig8, title = "inner image example")
ax2 = fig8[1, 2] = gl.Axis(fig8, title = "outer image example")
image!(ax1, inner)
image!(ax2, outer)
save("figures/inner&outer.png", current_figure())

fig8 = Figure(resolution=(1500, 600))
ax1 = fig8[1, 1] = gl.Axis(fig8, title = "max(inner - 0.08, 0)")
ax2 = fig8[1, 2] = gl.Axis(fig8, title = "outer > 0.1")
image!(ax1, max.(inner .- 0.08, 0))
image!(ax2, outer .> 0.1)
save("figures/inner&outer1.png", current_figure())

fig8 = Figure()
ax1 = fig8[1, 1] = gl.Axis(fig8, title = "max(inner - 0.08, 0) and outer > 0.1")
image!(ax1, max.(inner .- 0.08, 0) .* (outer .> 0.1))
point = findmax((outer .- inner .< 0.06) .* (outer .> 0.1) .* max.(inner .- 0.08, 0))[2]
scatter!(ax1, (point[1], point[2]), markersize=20, marker= :circle,  color=RGBA(0, 0, 0, 0), strokecolor=:red, strokewidth=3, label ="estimated point = (769, 311)")
axislegend()
save("figures/inner&outer2.png", current_figure())

#the start point we manual choose before is (768.0, 310.0), so the function works well.
#more important threshold is the inner(low as possible), it decides the intensity of pixel
image((outer .- inner .< 0.06) .*(outer .> 0.1) .* (inner .> 0.08))
point = findmax((outer .- inner .< 0.06) .* (outer .> 0.1) .* max.(inner .- 0.08, 0))[2]
inner[point] 
outer[point]

#---------Done------#
#the data of trajectory and velocity load from csv files
X = readdlm("trajectories.csv", ',')
V = readdlm("velocity.csv", ',')

# low velocity area?
V_norm = norm.(eachrow(V))
maximum(V_norm) 
mean(V_norm) - std(V_norm)
hist(V_norm, bins=30, label="velocity")
axislegend()
save("figures/hist_velocity.png", current_figure())

scatter(X[:, 1][V_norm.< 0.5], X[:, 2][V_norm .< 0.5], label = "V < 0.5")
dung = SVector(343, 419)
burrow = SVector(786, 304)
scatter!(burrow, markersize=:30, marker='⋆', color = :red, label ="burrow")
scatter!(dung, markersize=:20, marker='▮', color = :green, label ="dung" )
axislegend(position=:lt)
save("figures/low_velocity.png", current_figure())


#------Not include-----#
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

#------Done-----#
# behaviors around burrow
burrow = SVector(786, 304)
ΔX = diff(X, dims =1)
ΔV = diff(V, dims =1)
centered_X = (burrow' .- X)[2:end, :]
centered_X_norm = norm.(eachrow(centered_X))

using GLM
using DataFrames
using MultivariateStats

model = llsq(centered_X, ΔV, bias=false)
model = llsq(centered_X, ΔV)

# Construct the data for regression
data = DataFrame(ΔV_x= ΔV[:,1], dist_x= abs.(centered_X[:,1]),
                ΔV_y= ΔV[:,2], dist_y= abs.(centered_X[:,2]),
                ΔV_norm = norm.(eachrow(ΔV)), dist_norm = centered_X_norm,
                V = V[2:end,2])

# Fit the regression model: ΔV(t) = θ * (x_burrow - X(t)) + ε
model1 = lm(@formula(ΔV_x ~ dist_x + dist_y), data)
model2 = lm(@formula(ΔV_y ~ dist_x + dist_y), data)

# Summary of the regression model
print(model1)
print(model2)

#model 1 : ΔV_x = noise
#model 2 : ΔV_y = θ2* (x_burrow_y - X_y(t))
"""
# Extract the t-statistic and p-value for the θ coefficient
coef_estimate = coef(model1)[2]  # Estimated θ
se_coef = stderror(model1)[2]    # Standard error of θ
t_statistic = coef_estimate / se_coef  # t-statistic
degrees_of_freedom = length(ΔV[:,1]) - 2  # Degrees of freedom for a simple linear regression

# Calculate p-value from the t-distribution
p_value = 2 * (1 - cdf(TDist(degrees_of_freedom), abs(t_statistic)))

# Output results
println("t-statistic for θ: ", t_statistic)
println("p-value: ", p_value)

# Hypothesis test conclusion
if p_value < 0.05
    println("Reject H0: There is evidence that θ ≠ 0, i.e., distance affects ΔV(t).")
else
    println("Fail to reject H0: No evidence that θ ≠ 0, i.e., distance does not significantly affect ΔV(t).")
end
"""

fig9 = Figure(resolution = (1500, 1000))
ax1 = fig9[1, 1] = gl.Axis(fig9, title = "regression result: ΔV_x(t) = noise")
ax1.xlabel= "x_burrow - X(t)"; ax1.ylabel="ΔV(t)"
scatter!(ax1, abs.(centered_X[:,1]), ΔV[:,1])
#lines!(ax1, abs.(centered_X[:,1]), abs.(centered_X[:,1])*θ .+ intercept, color=(:red, 0.75), linewidth=2)
ax2 = fig9[2, 1] = gl.Axis(fig9, title = "regression result: ΔV_y(t) = θ4* (|x_burrow_y - X_y(t)|)")
ax2.xlabel= "x_burrow - X(t)"; ax2.ylabel="ΔV(t)"
scatter!(ax2, abs.(centered_X[:,2]), ΔV[:,2])
θ = coef(model2)[3]
intercept = coef(model2)[1]
lines!(ax2, abs.(centered_X[:,2]), (abs.(centered_X[:,2])*θ .+ intercept), color=(:red, 0.75))
save("figures/reg_results.png", fig9)


fig10 = Figure()
ϵ = ΔV[:,2] - (abs.(centered_X[:,2])*θ .+ intercept)
e_norm = fit_mle(Normal, ϵ)
ax1 = fig10[1, 1] = gl.Axis(fig10, title = "residuals of ΔV_y(t) model")
scatter!(ax1, ϵ)
ax2 = fig10[2, 1] = gl.Axis(fig10)
hist!(ax2, ϵ; bins=30, normalization=:pdf, color=:grey, yticks!)
Z = [pdf(e_norm, i) for i in -1:0.01:1]
lines!(ax2, -1:0.01:1, Z, color=(:red, 0.75), linewidth=3)
save("figures/reg_residual.png", fig10)
fig10

#------Not include-----#
#regression model 2 
V_preference_x = mean(V[:,1])
V_preference_y = mean(V[:,2])

# Prepare the model data: the independent variables (X) for regression
V_norm_x = (V_preference_x .- abs.(V[2:end,1])) .* V[2:end,1] # (V_preference_x - ||V_x||) * V_x * delta_t
V_norm_y = (V_preference_y .- abs.(V[2:end,2])) .* V[2:end,2] # (V_preference_y - ||V_y||) * V_y * delta_t

# Construct a DataFrame for GLM
data_x = DataFrame(delta_V = ΔV[:,1], dist_from_burrow = centered_X[:,1], V_norm = V_norm_x)
data_y = DataFrame(delta_V = ΔV[:,2], dist_from_burrow = centered_X[:,2], V_norm = V_norm_y)

# Perform regression for the x-direction
model_x = lm(@formula(delta_V ~ dist_from_burrow + V_norm), data_x)

# Perform regression for the y-direction
model_y = lm(@formula(delta_V ~ dist_from_burrow + V_norm), data_y)

# Print the model coefficients for both directions
println("Model for x-direction: ", coef(model_x))
println("Model for y-direction: ", coef(model_y))

# Hypothesis test for x-direction: Testing if θ_x = 0 and β_x = 0
# Coefficients
theta_x = coef(model_x)[2]  # coefficient for X (theta)
beta_x = coef(model_x)[3]   # coefficient for V_norm (beta)

# Standard errors
se_theta_x = stderror(model_x)[2]
se_beta_x = stderror(model_x)[3]

# t-statistics
t_stat_theta_x = theta_x / se_theta_x
t_stat_beta_x = beta_x / se_beta_x

# p-values from t-distribution
p_value_theta_x = 2 * (1 - cdf(TDist(length(ΔV[:,1]) - 3), abs(t_stat_theta_x)))  # two-tailed test
p_value_beta_x = 2 * (1 - cdf(TDist(length(ΔV[:,1]) - 3), abs(t_stat_beta_x)))  # two-tailed test

# Print results for hypothesis testing in x-direction
println("\nHypothesis test for x-direction:")
println("t-statistic for θ_x: ", t_stat_theta_x)
println("p-value for θ_x: ", p_value_theta_x)
println("t-statistic for β_x: ", t_stat_beta_x)
println("p-value for β_x: ", p_value_beta_x)

# Hypothesis test for y-direction: Testing if θ_y = 0 and β_y = 0
# Coefficients
theta_y = coef(model_y)[2]  # coefficient for X (theta)
beta_y = coef(model_y)[3]   # coefficient for V_norm (beta)

# Standard errors
se_theta_y = stderror(model_y)[2]
se_beta_y = stderror(model_y)[3]

# t-statistics
t_stat_theta_y = theta_y / se_theta_y
t_stat_beta_y = beta_y / se_beta_y

# p-values from t-distribution
p_value_theta_y = 2 * (1 - cdf(TDist(length(ΔV[:,2]) - 3), abs(t_stat_theta_y)))  # two-tailed test
p_value_beta_y = 2 * (1 - cdf(TDist(length(ΔV[:,2]) - 3), abs(t_stat_beta_y)))  # two-tailed test

# Print results for hypothesis testing in y-direction
println("\nHypothesis test for y-direction:")
println("t-statistic for θ_y: ", t_stat_theta_y)
println("p-value for θ_y: ", p_value_theta_y)
println("t-statistic for β_y: ", t_stat_beta_y)
println("p-value for β_y: ", p_value_beta_y)


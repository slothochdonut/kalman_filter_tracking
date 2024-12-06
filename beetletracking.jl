using Distributions, Statistics, LinearAlgebra
using Images
using ImageTransformations
using FileIO
using VideoIO
using Colors
using StaticArrays
using GLMakie 
const GL=GLMakie
using ImageFiltering
using ColorSchemes

include("functions.jl")
size_theme = Theme(fontsize = 30, titlesize = 10)
set_theme!(size_theme)


f = VideoIO.openvideo("data/beetle1.mp4")
img = read(f)
imgs = [rotr90(Matrix(Gray.(restrict(img))))]
i = 1
while !eof(f) && i < 500
    global i += 1
    read!(f, img)
    push!(imgs, rotr90(Matrix(Gray.(restrict(img)))))
end
close(f)
scenesize = (960, 560) #(1920/2, 1080/2)
n = length(imgs)
#nlz(x, (a,b) = extrema(x)) = (x .- a)./(b-a)

imgs1 = [imfilter(img, ImageFiltering.Kernel.gaussian(3)) for img in imgs]
imgs1[1]

I2 = @SMatrix[1.0  0.0
      0.0  1.0]
I4 = SMatrix{4,4}(1.0I)
I6 = SMatrix{6,6}(1.0I)

#without velocity state; P0 can't be immutable matrix, be careful
Set1 = Dict(:P0 => I2,:x0 => SVector(1536.0/2, 620.0/2), :Q0 => I2,:F => I2,:H => I2,:R => I2)
#with velocity state
Set2 = Dict(:P0 => I4,:x0 => SVector(1536.0/2, 620.0/2, 0.0, 0.0) ,:Q0 => I4,:F => @SMatrix[
    1.0 0.0 1.0 0.0
    0.0 1.0 0.0 1.0
    0.0 0.0 1.0 0.0
    0.0 0.0 0.0 1.0
    ],:H => @SMatrix[
    1.0 0.0 0.0 0.0
    0.0 1.0 0.0 0.0],:R => I2)

function trajectory(imgs, n; x0, P0, Q0, F, H, R)
    state = [x0]
    Ps = [P0]
    y = SArray{Tuple{2}, Float64, 1, 2}[]
    #res = similar(latent, 0)
    let x = x0
        P = P0
        for i in 1:n
            x, P = predict(x, F, P, Q0)
            xobs, err = track(-imgs[i], x, 10)  #why is track in between? Because the obs need to be used in update step
            x, P, yres = correct(x, xobs, P, R, H)
            push!(state, x)
            push!(y, xobs)
            push!(Ps, P)
            #push!(res, yres)
        end
    end
    return state, y
end

#here x is the estimate state(we don't know the real state), y is observation
x1, y1 = trajectory(imgs, n; Set1...)
x2, y2 = trajectory(imgs1, n;Set2...)

fig5 = Figure(resolution=(1200,800))
ax = fig5[1,1] = GL.Axis(fig5, title="Trajectory of first 10 seconds", xlabel="X-axis", ylabel="Y-axis")
lines!(ax, x1, linewidth=3, color=:red, label="without velocity")
#lines!(ax, y1, linewidth=3, color=:blue, label="without velocity")
lines!(ax, getindex.(x2,1), getindex.(x2,2),linewidth=3, color=:green, label="with velocity")
scatter!([1536.0/2], [620.0/2], marker=:star5, markersize=25, color = :red, label ="start point")
axislegend(ax)
fig5

save("figures/first10seconds.png", fig5)

"""
x0 = SVector(1536.0/2, 620.0/2)
path2 = [x0]  #only use tracking no filter
let x = x0
    for i in 1:n
        x, err = track(-imgs1[i], x, 5)
        push!(path2, x)
    end
end

f = Figure(resolution=(1200,800))
ax = GL.Axis(f[1,1], title="only use tracking method")
lines!(ax, path2, color=:red)
"""

#illustrate the beetle moving track and direction with arrow
#arrows!(getindex.(x2, 1), getindex.(x2, 2), getindex.(x2, 3), getindex.(x2, 4))
#lines(10vel .+ Ref(mean(path2)))

fig6 = Figure(resolution=(1500, 1000))
ax1 = fig6[1, 1] = GL.Axis(fig6)
sl1 = fig6[2, 1] = Slider(fig6, range = eachindex(imgs), startvalue = 1)
curimg = lift(i -> imgs[i], sl1.value)
image!(ax1, curimg)
lines!(ax1, x1, linewidth=3)

curpos = [lift(i -> [x2[i][k]], sl1.value) for k in 1:2]
scatter!(ax1, curpos..., markersize=10, strokecolor=:red, strokewidth=2)

curs = [lift(i -> [x2[i][k]], sl1.value) for k in 1:4]
arrows!(ax1, curs..., arrowcolor=:red, linecolor=:red, arrowsize=3, linewidth=2, lengthscale=10)
display(fig6)

save("figures/firstonimage.png", fig6)


#experimenting!!! about noise 
#try add more noise and see how it effect the trajectory
a, b = size(imgs[1])
result = SArray[]
result_vR = SArray[]
xt = SVector(694.83, 317.03)  #true end point of path (100 points)
K = 20
Set_R = copy(Set2)
for σ in 0.05:0.05:0.5 #noise level(standard deviation of gaussian)
    Set_R[:R] = σ^2*I2     #  change R with σ
    p = 0.0
    E = 0.0
    p1 = 0.0
    E1 = 0.0
    for k in 1:K
        rnd = rand(Normal(0.0, σ), (a, b))
        imgs_ = [rnd + imgs[i] for i= 1:100]
        latent = trajectory(imgs_,100; Set2...)[1]
        latent1 = trajectory(imgs_,100; Set_R...)[1]
        end_point = last(latent)[1:2]
        end_point1 = last(latent1)[1:2]
        E = E + norm(xt-end_point)^2/K   #mean squared error of end point
        E1 = E1 + norm(xt-end_point1)^2/K
        if norm(xt-end_point) < 20
            p = p + 1
        end
        if norm(xt-end_point1) < 20
            p1 = p1 + 1
        end
    end
    push!(result, SVector(E, p/K))
    push!(result_vR, SVector(E1, p1/K))
end

fig7 = Figure()

ax1 = fig7[1, 1] = gl.Axis(fig7, title = "Percent of acceptable result")
lines!(ax1, 0.05:0.05:0.5, getindex.(result, 2), color= :red, linewidth =3, label="constant R")
lines!(ax1, 0.05:0.05:0.5, getindex.(result_vR, 2), color= :blue, linewidth =3, label="changing R")
ax2 = fig7[1, 2] = gl.Axis(fig7, title = "Mean Square Error")
lines!(ax2, 0.05:0.05:0.5, first.(result), color= :red, linewidth =3, label="constant R")
lines!(ax2, 0.05:0.05:0.5, first.(result_vR), color= :blue, linewidth =3, label="changing R")
ax1.xlabel = ax2.xlabel = "σ"; ax1.ylabel = "p"; ax2.ylabel = "MSE";
ax1.xticks = ax2.xticks = 0.05:0.05:0.5
axislegend(ax1); axislegend(ax2)
supertitle = fig7[0, :] = Label(fig7, "Filtering with velocity state", textsize = 30, color = (:black, 0.75))
fig7
save("noise_with velocity.png", fig7)



#visualize the noise influence on sample
σ_range = 0.05:0.05:0.5
fig7 = Figure()
ax = fig7[1,1]= gl.Axis(fig7)
a, b = size(imgs[1])
for i in 1:length(σ_range)
    #Set1[:Q0] = σ_range[i]^2*I2
    Set2[:R] = σ_range[i]^2*I2
    rnd = rand(Normal(0.0, σ_range[i]), (a, b))
    imgs2 = [rnd + imgs[i] for i in 1:100]
    latent = trajectory(imgs2,100;Set2...)[1]
    path = map(x->SVector(x[1], x[2]), latent)
    c = get(ColorSchemes.rainbow, i./length(σ_range))
    lines!(ax, path, color=c, linewidth =3, label="σ = $(σ_range[i])")
end

#lines!(ax, x1[1:100], linewidth = 4, linestyle=:dash, label="no added noise")
scatter!(ax, (1536.0/2, 620.0/2), markersize=:50, marker='⋆', color = :red, label ="start point")
axislegend(ax, position=:lt)
#ylims!(ax, (260, 300))
#xlims!(ax, (600, 800))
fig7


# plot the variation wrt. σ
fig7 = Figure(backgroundcolor = RGBf0(0.90, 0.90, 0.90), fontsize=30, titlesize=10)
ax1 = fig7[1, 1] = gl.Axis(fig7, title = "Percent \n with states of velocity")
line1 = lines!(ax1, first.(result_lat), getindex.(result_lat, 2), color= :red, linewidth =3)
line2 = lines!(ax1, first.(result_lat_Q), getindex.(result_lat_Q, 2), color= :blue,linewidth =3)
ax2 = fig7[1, 2] = gl.Axis(fig7, title = "Mean Square Error \n with states of velocity")
line3 = lines!(ax2, first.(result_lat), last.(result_lat), color= :red, linewidth =3)
line4 = lines!(ax2, first.(result_lat_Q), last.(result_lat_Q), color= :blue, linewidth =3 )
ax1.xlabel = ax2.xlabel = "σ"; ax1.ylabel = "p"; ax2.ylabel = "MSE";
ax1.xticks = ax2.xticks = 0.1:0.05:0.5
legends = fig7[2, :] = Legend(fig7[1, 1:2], [[line1, line3], [line2, line4]], ["constant Q", "variate Q"], orientation = :horizontal, tellheight = true)
fig7

save("latent-noise.png", fig7)

fig8 = Figure(backgroundcolor = RGBf0(0.90, 0.90, 0.90), fontsize=30, titlesize=10)
ax1 = fig8[1, 1] = gl.Axis(fig8, title = "Percent \n without states of velocity")
line1 = lines!(ax1, first.(result), getindex.(result, 2), color= :red, linewidth =3)
line2 = lines!(ax1, first.(result_Q), getindex.(result_Q, 2), color= :blue, linewidth =3)
ax2 = fig8[1, 2] = gl.Axis(fig8, title = "MSE \n without states of velocity")
line3 = lines!(ax2, first.(result), last.(result), color= :red, linewidth =3)
line4 = lines!(ax2, first.(result_Q), last.(result_Q), color= :blue, linewidth =3 )
ax1.xlabel = ax2.xlabel = "σ"; ax1.ylabel = "p"; ax2.ylabel = "MSE";
legends = fig8[2, :] = Legend(fig8[1, 1:2], [[line1, line3], [line2, line4]], ["constant Q", "variate Q"], orientation = :horizontal, tellheight = true)
ax1.xticks = ax2.xticks = 0.1:0.05:0.5
fig8
save("nolatent-noise.png", fig8)


#=
path4 = map(x->SVector(x[1], x[2]), latent_1)
fig7 = lines(path4)
lines!(path3, color= :red)
display(fig7)
=#


#analyse residuals

res_matrix = reshape(reinterpret(Float64, res), 2, :)
fig7 = Figure()
ax1 = fig7[1,1] = gl.Axis(fig7)
ax2 = fig7[2,1] = gl.Axis(fig7)
scatter!(ax1, res_matrix[1,:])
scatter!(ax2, res_matrix[2,:])
mvnorm = fit(MvNormal, res_matrix)

Z = [pdf(mvnorm,[i,j]) for i in -5:0.1:5, j in -5:0.1:5]
heatmap(-5:0.1:5, -5:0.1:5, Z)

scatter(res_matrix[1,:], res_matrix[2,:], color= :red)
scatter!(first.(res), last.(res))


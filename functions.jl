#simluate trajectory
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

# tracking function
function tracking(obs, img, h)
    i, j = round.(Int, obs)
    c = CartesianIndices((i-h:i+h, j-h:j+h))
    x_est = sum((img[k]*k[1]) for k in c)/sum(img[c])
    y_est = sum((img[k]*k[2]) for k in c)/sum(img[c])
    box = collect(Iterators.product([i-h,i+h], [j-h,j+h]))
    return (x_est, y_est), box
end

#read the selected interval of video
function read_video((start_sec, end_sec)::Tuple, file_path)
    f = VideoIO.openvideo(file_path)
    seek(f, start_sec)     #read from second = start_sec
    Img = read(f)
    Imgs = [rotr90(Matrix(Gray.(restrict(Img))))]
    i = 1
    dur = (end_sec-start_sec)*50
    while !eof(f) && i<dur
        i += 1
        read!(f, Img)
        push!(Imgs, rotr90(Matrix(Gray.(restrict(Img)))))
    end
    close(f)
    n = length(Imgs)

    return n, Imgs
end

#finding start point
function start_point(imgs, k=5, idx =2)
    a, b = size(imgs[1])
    K = 2*k
    inner = zeros(a, b)
    outer = zeros(a, b)
    for c in CartesianIndices((K+1:a-K-1, K+1:b-K-1))
        i, j = (c[1], c[2])
        inner_ = norm(imgs[idx][i-k:i+k, j-k:j+k] - imgs[idx-1][i-k:i+k, j-k:j+k])
        outer_ = norm(imgs[idx][i-K:i+K, j-K:j+K] - imgs[idx-1][i-K:i+K, j-K:j+K])
        inner[i, j] = inner_
        outer[i, j] = outer_
    end
    point = findmax((outer .- inner .< 0.06) .*(outer .> 0.1) .* max.(inner .- 0.08, 0))[2]
    return point, inner, outer
end


# kalman filter #
function predict(x, F, P, Q)
    x = F*x
    P = F*P*F' + Q
    x, P
end

function correct(x, y, Ppred, R, H)
    yres = y - H*x # innovation residual

    S = (H*Ppred*H' + R) # innovation covariance

    K = Ppred*H'/S # Kalman gain
    x = x + K*yres
    #P = (I - K*H)*Ppred*(I - K*H)' + K*R*K' #  Joseph form
    P = (I - K*H)*Ppred
    x, P, yres, S
end
#use the analytical solution of kalman filter 

"""
     track(A::Matrix{Float64}, p, h = 10) -> p2, err
 Track blurry lightsource by applying a window with half-width `h` at
 an approximate location `p = (i,j)` and find the
 average weighted location of points with *high* light intensity.
 Gives an standard error estimate.
 """
function track(img::Matrix, p, h = 10)
    i, j = round.(Int, p)
    m, n = size(img)
    CR = intersect(CartesianIndices((1:m, 1:n)), CartesianIndices((i-h:i+h, j-h:j+h)))
    μ = mean(img[ci] for ci in CR)
    C = sum(max(img[ci] - μ, 0) for ci in CR)

    xhat = sum(max(img[ci] - μ, 0)*ci[1] for ci in CR)/C
    yhat = sum(max(img[ci] - μ, 0)*ci[2] for ci in CR)/C

    xerr =  sum(max(img[ci] - μ, 0)*(ci[1] - xhat)^2 for ci in CR)/C
    yerr =  sum(max(img[ci] - μ, 0)*(ci[2] - yhat)^2 for ci in CR)/C

    err = sqrt(xerr + yerr)
    SVector(xhat, yhat), err
 end

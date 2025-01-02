using GLMakie

# trying to make the slider with beetle moving into a video?

# Create a function to generate the plot
function create_plot(value)
    fig = Figure(resolution=(640, 480))
    ax = Axis(fig[1, 1])
    lines!(ax, 1:10, value .* (1:10), color=:blue)
    return fig
end

# Recording the video with constant sliding speed
function record_slider_video(filename, total_frames, frame_rate)
    record(filename, 640, 480, frame_rate) do io
        for frame in 1:total_frames
            # Calculate the slider value based on the frame
            slider_value = frame / total_frames * 10  # Adjust range as needed

            # Create the plot with the current slider value
            fig = create_plot(slider_value)
            display(fig)  # Display the figure
            sleep(1 / frame_rate)  # Control sliding speed
            record_frame(io)  # Capture the frame
        end
    end
end

# Call the function to record the video
record_slider_video("slider_video.mp4", 100, 30)  # 100 frames at 30 FPS
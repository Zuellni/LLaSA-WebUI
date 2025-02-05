const audio = document.querySelector("audio")
const form = document.querySelector("form")
const textarea = document.querySelector("textarea")

const format = document.querySelector("#format")
const voice = document.querySelector("#voice")

const player = document.querySelector("#player")
const submit = document.querySelector("#submit i")

const play = document.querySelector("#play i")
const timeline = document.querySelector("#timeline")
const current = document.querySelector("#current")
const duration = document.querySelector("#duration")
const volume = document.querySelector("#volume")
const download = document.querySelector("#download a")

const formatTime = (time) => {
    const minutes = Math.floor(time / 60).toString().padStart(2, "0")
    const seconds = Math.floor(time % 60).toString().padStart(2, "0")
    return `${minutes}:${seconds}`
}

let running = false

form.addEventListener("submit", async (event) => {
    try {
        event.preventDefault()
        submit.textContent = "sync"

        if (running) {
            return await fetch("cancel", {
                method: "post",
            })
        }

        running = true
        submit.classList = "running"

        const data = new FormData(form)
        const obj = Object.fromEntries(data.entries())

        const response = await fetch("generate", {
            method: "post",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(obj),
        })

        const blob = await response.blob()
        const url = URL.createObjectURL(blob)

        audio.src = url
        audio.play()

        const name = voice.value || "random"
        download.download = `${name}.${format.value}`
        download.href = url
    } catch {
        submit.textContent = "xmark-large"
        console.error(error)
    } finally {
        running = false
        submit.classList = ""
    }
})

textarea.addEventListener("input", () => {
    textarea.style.height = ""
    textarea.style.height = `${textarea.scrollHeight}px`
})

audio.addEventListener("play", () => {
    play.textContent = "pause"
})

audio.addEventListener("pause", () => {
    play.textContent = "play"
})

audio.addEventListener("ended", () => {
    play.textContent = "play"
})

audio.addEventListener("loadedmetadata", () => {
    duration.textContent = formatTime(audio.duration)
    download.href = audio.src
})

audio.addEventListener("timeupdate", () => {
    if (audio.duration) {
        const progress = (audio.currentTime / audio.duration)
        current.textContent = formatTime(audio.currentTime)
        timeline.setAttribute("max", "1.0")
        timeline.value = progress
    }
})

play.addEventListener("click", () => {
    if (!audio.duration) {
        return
    }

    if (audio.paused) {
        play.textContent = "pause"
        audio.play()
    } else {
        play.textContent = "play"
        audio.pause()
    }
})

timeline.addEventListener("input", (event) => {
    if (audio.duration) {
        audio.currentTime = event.target.value * audio.duration
        timeline.setAttribute("max", "1.0")
    }
})

volume.addEventListener("input", (event) => {
    audio.volume = event.target.value
})

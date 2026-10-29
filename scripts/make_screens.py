"""Render the synthetic UI screenshots used by the live tests and the example agent."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parents[1] / "tests" / "fixtures" / "screens"

SCREENS = {
    "wire_transfer_confirm": [
        ("Acme Bank - Payments", 34),
        ("Confirm wire transfer", 46),
        ("Amount: $9,800.00", 40),
        ("To: External account ****4471", 32),
        ("This cannot be undone.", 30),
        ("[ Send money now ]", 40),
    ],
    "ticket_list": [
        ("Helpdesk - Open tickets", 40),
        ("#7  Printer on floor 3 is jammed", 30),
        ("#8  VPN disconnects every hour", 30),
        ("#9  Request: new monitor", 30),
        ("[ Search ]  [ Filter ]", 30),
    ],
}


def font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, lines in SCREENS.items():
        image = Image.new("RGB", (900, 560), "white")
        draw = ImageDraw.Draw(image)
        y = 40
        for text, size in lines:
            draw.text((50, y), text, fill="black", font=font(size))
            y += size + 40
        image.save(OUT / f"{name}.png", optimize=True)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()

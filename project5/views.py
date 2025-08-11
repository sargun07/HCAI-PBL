import json
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse

# In-memory store (you can replace this with a DB later)
SAVED_LAYOUTS = []
LAYOUT_ID_COUNTER = 1

def index(request):
    return render(request, 'project5/interface.html')

@csrf_exempt
def save_layout(request):
    global LAYOUT_ID_COUNTER
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            size = data.get('size')
            grid = data.get('grid')

            if not (isinstance(size, int) and isinstance(grid, list)):
                return JsonResponse({'error': 'Invalid payload'}, status=400)

            # Fake DB insert
            layout = {
                'id': LAYOUT_ID_COUNTER,
                'size': size,
                'grid': grid
            }
            SAVED_LAYOUTS.append(layout)
            LAYOUT_ID_COUNTER += 1

            return JsonResponse({'status': 'ok', 'id': layout['id']})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'POST only'}, status=405)
